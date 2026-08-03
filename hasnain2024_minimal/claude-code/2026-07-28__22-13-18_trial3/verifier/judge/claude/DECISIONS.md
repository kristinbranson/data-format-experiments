# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB file, `data_structure_<anm>_<date>.mat`, located in either `Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior` subdirectories. Sessions are hard-coded in `EPHYS_SESSIONS` as tuples of (animal, date, probes, data_dir). The AI checks file existence before loading and skips missing files. Data loading uses a `SessionData` class that detects v7.3 HDF5 vs v5 MATLAB format by reading the file header. Motion energy is loaded separately from `motionEnergy_<anm>_<date>.mat` files.

ii.
```python
EPHYS_SESSIONS = [
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ...
    ('JEB24', '2023-11-03', [1], 'RandomizedDelay_Ephys_Behavior'),
]

for anm, date, probes, data_dir in available:
    result = process_session(anm, date, probes, data_dir, time_edges, time_axis)
```

```python
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

iii. The AI's CONVERSION_NOTES.md states sessions were selected from the authors' loading scripts. The AI includes 45 sessions (8 JEB23 sessions including 2023-10-20, vs 7 in the reference which excludes 2023-10-20). Two JEB24 sessions (2023-10-03, 2023-10-04) were excluded because they lack cluster data.

## 1-b. How are the data split into subjects?

i. The animal name is the first element of the session tuple (e.g., `'JEB19'`). Subject list is built by appending unique animal names in order of first appearance (not sorted). `subject_idx` maps each session to its subject index.

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

iii. Subject extraction from the session tuple is straightforward. The subjects list preserves insertion order rather than being sorted alphabetically.

## 1-c. How are the data split into sessions?

i. Each entry in `EPHYS_SESSIONS` defines one session. The AI lists 45 sessions total (25 Ephys_Behavior + 20 RandomizedDelay). Sessions whose files are not found on disk are skipped. Each session becomes one element of `neural`, `input`, and `output`.

ii.
```python
available = []
for anm, date, probes, data_dir in EPHYS_SESSIONS:
    fn = f'data_structure_{anm}_{date}.mat'
    fpath = os.path.join(DATA_ROOT, data_dir, fn)
    if os.path.exists(fpath):
        available.append((anm, date, probes, data_dir))
```

iii. The AI notes 45 sessions processed vs 44 in the reference, due to including JEB23_2023-10-20. The CONVERSION_NOTES.md says "Total: 45 sessions, 14 subjects, 2,504 units."

## 1-d. How are the data split into trials?

i. Trials are indexed from `bp.Ntrials`. Trial arrays (hit, miss, R, L, early, autowater, stim.enable) are loaded and truncated to `ntrials`. Each trial is one entry in these arrays, indexed by position.

ii.
```python
ntrials = sd.get_ntrials()
hit = sd.get_trial_array('hit').astype(bool)
miss = sd.get_trial_array('miss').astype(bool)
...
valid_idx = np.where(valid_trials)[0]
```

iii. Trial splitting follows the standard approach from the Bpod data structure.

## 1-e. How are trials filtered based on quality controls?

i. Three filters: (1) exclude early-lick trials, (2) exclude photostimulation trials, (3) **exclude ignore/no-response trials** -- only hit or miss trials are kept. The AI requires `(hit | miss) & ~stim_enable & ~early`. Sessions with fewer than 5 valid trials are skipped. Unlike the reference, the AI does NOT check whether trials extend past the end of the recording.

ii.
```python
valid_trials = (hit | miss) & ~stim_enable & ~early
valid_idx = np.where(valid_trials)[0]

if len(valid_idx) < 5:
    print(f'  WARNING: Too few valid trials ({len(valid_idx)}), skipping')
    sd.close()
    return None
```

iii. The CONVERSION_NOTES.md states: "Include: hit (correct) and miss (incorrect) trials. Exclude: stim-enabled trials, early lick trials, ignore/no-response trials." This explicitly drops ignore trials, unlike the reference which keeps them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu` cluster data: for each cluster, `quality`, `trial` (1-based trial number of each spike), and `trialtm` (spike time relative to trial start). Also `bp.ev.goCue` for alignment.

ii.
```python
clusters = sd.get_clusters(probe_nums)
# Each cluster is (quality, trial_array, trialtm_array)
aligned_times = spike_times[trial_mask] - gocue[j]
counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. Same raw variables as the reference.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into **10 ms bins** (DT=1/100) from -2.5 to 2.5 s (500 bins), converted to firing rates by dividing by DT, then smoothed with a **causal Gaussian kernel** (window=15 bins, first half zeroed, std=(15-1)/(2*2.5)=2.8 bins, reflect boundary). Spike binning is done per-cluster per-trial in a nested loop.

ii.
```python
DT = 1/100  # 10 ms
SMOOTH_WINDOW = 15

# Per-cluster, per-trial loop:
aligned_times = spike_times[trial_mask] - gocue[j]
counts, _ = np.histogram(aligned_times, bins=time_edges)
fr = counts / DT
trialdat[i, :, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

```python
def causal_gaussian_smooth(x, N, bctype='reflect'):
    kern = windows.gaussian(N, std=(N-1)/(2*2.5))
    kern[:N//2] = 0  # make causal
    kern = kern / kern.sum()
    ...
```

iii. The AI's CONVERSION_NOTES.md states "Bin size: 10 ms (dt = 1/100)" and "Smoothing: Causal Gaussian kernel (window = 15 bins)." The reference uses 5 ms bins and a symmetric (non-causal) Gaussian with sigma=14 ms.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, clusters with quality labels in `{'garbage', 'gabrga', 'noisy', 'real?'}` are excluded. Note: `poor` is NOT excluded (unlike the reference which excludes it). Then units with mean firing rate <= 1 Hz are removed. Sessions with fewer than 10 remaining units are skipped.

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR_THRESH = 1.0

filtered_clusters = [(q, t, tt) for q, t, tt in clusters if q not in EXCLUDE_QUALITIES]
...
mean_frs = np.mean(np.mean(trialdat[:, :, valid_idx], axis=2), axis=1)
keep_units = mean_frs > LOW_FR_THRESH
```

iii. CONVERSION_NOTES.md states: "Include all clusters EXCEPT those with quality labels: garbage, gabrga, noisy, real?. This includes: excellent, great, good (single units), fair, poor, multi (multi-units)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For each trial, spike times (trialtm) are aligned by subtracting the go cue time: `aligned_times = spike_times[trial_mask] - gocue[j]`. This puts spikes in seconds relative to go cue onset.

ii.
```python
aligned_times = spike_times[trial_mask] - gocue[j]
counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. Same alignment approach as the reference: subtract go cue from trial-relative spike times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses **10 ms bins** (DT = 1/100), producing 500 time bins from -2.5 to 2.5 s. No rebinning is applied -- spikes are directly binned at 10 ms resolution.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1/100  # 10 ms
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
```

iii. The AI interpreted `params.dt = 1/200` from the reference code as `1/100`, resulting in 10 ms bins instead of the reference's 5 ms bins (1000 bins).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the bin center time axis, defined purely from the time window parameters. It is not derived from any raw data variable -- it is the center of each time bin.

ii.
```python
time_axis = time_edges[:-1] + DT / 2
input_data = time_axis.reshape(1, -1).copy()
```

iii. Same approach as the reference: the input is defined by the time binning grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as the center of each bin. No further processing.

ii.
```python
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis uses the same bin centers as the neural data, so they share the same temporal grid by construction.

ii.
```python
input_data = time_axis.reshape(1, -1).copy()
# Same time_edges used for spike binning:
counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. Same approach as the reference.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI uses `bp.R` directly. If R is true, lick direction is right (1); otherwise left (0). Since ignore trials are already filtered out (only hit|miss kept), there is no need to handle no-lick trials.

ii.
```python
R = sd.get_trial_array('R').astype(bool)
...
lick_dir = 1 if R[trial_idx] else 0
```

iii. CONVERSION_NOTES.md: "lick_direction: left=0, right=1 (per-trial, from bp.R)."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A direct mapping from the instructed side R: R=True -> right=1, R=False -> left=0. This is the instructed side, not the actual lick direction. On miss trials, the animal licked the opposite side, so R does not correctly represent lick direction for miss trials. The output has 2 classes (left, right), with no "no lick" class since ignore trials were excluded.

ii.
```python
lick_dir = 1 if R[trial_idx] else 0
```

iii. The AI treats R as lick direction directly, but on miss trials, the actual lick was to the opposite side. The reference correctly derives lick direction from the combination of R and hit/miss.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater`. Autowater=1 indicates WC context.

ii.
```python
autowater = sd.get_trial_array('autowater')
context = 0 if autowater[trial_idx] == 1 else 1
```

iii. Same as reference.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct mapping: autowater=1 -> WC=0, else DR=1.

ii.
```python
context = 0 if autowater[trial_idx] == 1 else 1
```

iii. Matches the reference and instruction specification (WC=0, DR=1).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit` only. If hit is true, outcome is correct (1); otherwise incorrect (0).

ii.
```python
hit = sd.get_trial_array('hit').astype(bool)
outcome = 1 if hit[trial_idx] else 0
```

iii. Since only hit|miss trials are kept, `not hit` is equivalent to `miss`, making this correct within its filtered subset.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Binary mapping: hit -> correct=1, not hit -> incorrect=0. Only 2 classes. The reference has 3 classes (correct, incorrect, ignore) because it keeps ignore trials.

ii.
```python
outcome = 1 if hit[trial_idx] else 0
```

iii. Matches instructions (incorrect=0, correct=1) within the AI's filtered trial set.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. DeepLabCut tracking from `obj.traj`, using **only the bottom camera** (`view=1`), feature `top_tongue`. The reference uses both side and bottom cameras.

ii.
```python
bottom_feats = sd.get_traj_feature_names(1)
for idx, name in enumerate(bottom_feats):
    if name == 'top_tongue' and tongue_feat_idx is None:
        tongue_feat_idx = idx
```

iii. CONVERSION_NOTES.md states: "Computed from bottom camera DLC tracking of `top_tongue` feature."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Raw x, y positions are differenced (no smoothing), velocity computed as `sqrt(dx^2 + dy^2) / dt` where dt=1/400. The velocity is then interpolated (np.interp) to the 10 ms time bins. No likelihood filtering is applied -- NaN positions from DLC are carried through. No normalization or per-view scaling is done (only one view used). The AI does NOT smooth positions before differencing, does NOT filter by likelihood threshold, and does NOT handle contiguous runs.

ii.
```python
ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
dt_vid = 1.0 / VIDEO_FPS
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
vel_t = ft[:-1] + dt_vid / 2
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. CONVERSION_NOTES.md: "Velocity = sqrt(dx^2 + dy^2) / dt at 400 Hz, interpolated to 10ms bins."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Discretized at session 50th percentile (nanpercentile). NaN values (bins outside video coverage) are mapped to class 0 (below threshold), not to a separate "not visible" class. Only 2 output classes.

ii.
```python
tongue_thresh = np.nanpercentile(all_tongue, 50)
tongue_disc = np.where(np.isnan(t['tongue_vel']), 0,
                      np.where(t['tongue_vel'] >= tongue_thresh, 1, 0)).astype(np.int64)
```

iii. The reference uses 3 classes (below=0, above=1, not visible=2). The AI maps NaN to class 0, conflating "not visible" with "below threshold."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Video offset computed from bitcode (matching `findVideoOffset.m`). Frame times corrected by subtracting offset and go cue. Tongue velocity is then interpolated (`np.interp`) onto the neural time axis.

ii.
```python
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The reference bins frames into time bins by averaging; the AI interpolates instead.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DLC tracking from **bottom camera** (`view=1`), feature `bottom_paw`. The reference uses `top_paw`.

ii.
```python
for idx, name in enumerate(bottom_feats):
    if name == 'bottom_paw' and paw_feat_idx is None:
        paw_feat_idx = idx
```

iii. CONVERSION_NOTES.md: "paw_velocity: Same as tongue but using `bottom_paw` feature." The reference uses `top_paw` because `bottom_paw` drops out during the delay epoch.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue: raw diff of x, y positions (no smoothing), velocity as sqrt(dx^2+dy^2)/dt, interpolated to time bins. No likelihood filtering.

ii.
```python
ft, x, y = sd.get_trial_traj(1, trial_idx, paw_feat_idx)
dt_vid = 1.0 / VIDEO_FPS
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
vel_t = ft[:-1] + dt_vid / 2
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. Same processing pipeline as tongue velocity.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: 50th percentile threshold, NaN -> class 0. Only 2 classes.

ii.
```python
paw_disc = np.where(np.isnan(t['paw_vel']), 0,
                   np.where(t['paw_vel'] >= paw_thresh, 1, 0)).astype(np.int64)
```

iii. Same as tongue velocity discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: video offset correction, then interpolation to time bins.

ii.
```python
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. Same alignment approach as tongue.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from separate `motionEnergy_<anm>_<date>.mat` files. Handles multiple file formats (struct with data/moveThresh, nested structs, bare cell arrays).

ii.
```python
def load_motion_energy(data_dir, anm, date):
    me_mat = sio.loadmat(me_path, squeeze_me=False, struct_as_record=True)
    me_var = me_mat['me']
    if me_var.dtype.names is not None and 'data' in me_var.dtype.names:
        ...
    if me_var.dtype == object:
        ...
```

iii. Same source as the reference.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values are interpolated (`np.interp`) from camera frame times onto the 10 ms bins. The reference averages frames within each bin instead.

ii.
```python
ft = sd.get_trial_frame_times(1, trial_idx)
ft_aligned = ft - vidshift - gocue[trial_idx]
me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. CONVERSION_NOTES.md: "Interpolated from 400 Hz video frame rate to 10ms time bins."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as tongue/paw: 50th percentile, NaN -> class 0. Only 2 classes.

ii.
```python
me_disc = np.where(np.isnan(t['motion_energy']), 0,
                  np.where(t['motion_energy'] >= me_thresh_50, 1, 0)).astype(np.int64)
```

iii. Same approach as other continuous outputs.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Uses **bottom camera** (`view=1`) frame times for alignment, corrected by video offset and go cue. The reference uses the **side camera** (`view=0`) since motion energy is computed from the side camera.

ii.
```python
ft = sd.get_trial_frame_times(1, trial_idx)  # view=1 is bottom camera
ft_aligned = ft - vidshift - gocue[trial_idx]
me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], ...)
```

iii. The AI uses `get_trial_frame_times(1, ...)` (bottom camera view=1) while motion energy is derived from the side camera, whose frame times may differ.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Errors in trajectory loading are caught by bare `except: pass` blocks, causing the affected trial's tongue/paw/motion-energy to remain as NaN arrays. NaN values are then mapped to class 0 (below threshold) during discretization. Trials with all-zero neural data (past recording end) are kept. Missing motion energy files cause `None` return, and all ME bins become NaN -> class 0.

ii.
```python
try:
    ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
    ...
except:
    pass
```

```python
tongue_disc = np.where(np.isnan(t['tongue_vel']), 0, ...)
```

iii. CONVERSION_NOTES.md: "Warnings: ~30 trials with all-zero neural data (late trials in sessions where recording quality degraded)." These are kept rather than dropped.

## 11-a. What are the most time-consuming steps of the code?

i. Two main bottlenecks: (1) Loading MATLAB files via `SessionData.__init__`, and (2) the nested loop over clusters and trials for spike binning, which iterates over every cluster and every trial individually.

ii.
```python
# Per-cluster, per-trial spike binning (nested loop)
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        trial_mask = (trial_nums == (j + 1))
        aligned_times = spike_times[trial_mask] - gocue[j]
        counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. The nested loop is significantly slower than the reference's vectorized `np.histogram2d` approach.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop over clusters and trials is the main candidate. The reference uses `np.histogram2d` to bin all trials at once for each cluster. The AI's nested `for j in range(ntrials)` loop with per-trial masking and histogram is much slower. The per-trial trajectory processing loops could also be partially vectorized.

ii.
```python
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        trial_mask = (trial_nums == (j + 1))
        ...
        counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. The reference's vectorized approach is significantly more efficient.

## 11-c. What processing does the code repeat multiple times?

i. The `get_trial_traj` method re-reads trajectory data from the HDF5/v5 structure for each feature (tongue, paw) and each trial separately, dereferencing HDF5 references each time. Frame times are also fetched separately for motion energy alignment, even though they were already read during tongue/paw processing.

ii.
```python
# Tongue: reads traj for each trial
ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
# Paw: reads traj again for each trial
ft, x, y = sd.get_trial_traj(1, trial_idx, paw_feat_idx)
# Motion energy: reads frame times a third time
ft = sd.get_trial_frame_times(1, trial_idx)
```

iii. Frame times for the bottom camera are read 3 times per trial (tongue, paw, motion energy).

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI bins spikes for ALL trials (including invalid ones) in `trialdat` before selecting valid trials. The full `trialdat` array is `(n_units, n_timebins, ntrials)` over all trials, but only `valid_idx` trials are used. Also, mean firing rate is computed over all time bins and valid trials but only used for the threshold check.

ii.
```python
trialdat = np.zeros((n_units, n_timebins, ntrials))
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):  # loops over ALL trials, not just valid ones
        ...
```

iii. Computing spike counts and smoothing for trials that will be discarded (early, stim, ignore) wastes computation.
