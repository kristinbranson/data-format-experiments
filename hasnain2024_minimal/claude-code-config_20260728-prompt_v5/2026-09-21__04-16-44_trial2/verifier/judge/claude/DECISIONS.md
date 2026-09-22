# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB file, `data_structure_<anm>_<date>.mat`, located in either `Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`. The 44 sessions and their probes are hard-coded in `SESSION_META`, transcribed from the authors' loading scripts. The AI auto-detects the file format: it first tries HDF5 (v7.3) via `load_session_h5`, and falls back to scipy (v5) via `load_session_scipy`. Motion energy is loaded separately from `motionEnergy_<anm>_<date>.mat` files.

ii.
```python
SESSION_META = [
    ('EKH1', '2021-08-07', [2], EPHYS_DIR),
    ...
    ('JEB24', '2023-11-03', [1], RANDDELAY_DIR),
]

def load_session(anm, date, probe_list, data_dir):
    fn = f'data_structure_{anm}_{date}.mat'
    fpath = os.path.join(data_dir, fn)
    try:
        sess = load_session_h5(fpath, probe_list)
    except Exception:
        sess = load_session_scipy(fpath, probe_list)
```

iii. The agent examined the authors' `load<ANM>_ALMVideo.m` scripts to determine which sessions and probes to include, and noted that both HDF5 and scipy formats are needed since data files come in both MATLAB v7.3 and v5 formats.

## 1-b. How are the data split into subjects?

i. The animal name is the first element of the session metadata tuple (e.g., `'EKH1'`). Subjects are accumulated in order of first appearance. `subject_idx` maps each session to its subject index.

ii.
```python
for anm, date, probes, data_dir in SESSION_META:
    ...
    if anm not in all_subjects:
        all_subjects.append(anm)
    subj_idx = all_subjects.index(anm)
```

iii. The agent extracted animal names from the session metadata, which mirrors the naming convention in the loading scripts.

## 1-c. How are the data split into sessions?

i. One session is one entry in `SESSION_META`. The data directory is specified per session (either `EPHYS_DIR` or `RANDDELAY_DIR`). Each becomes one element of the `neural`, `input`, and `output` lists. Sessions with fewer than `MIN_UNITS=10` good units after filtering are skipped entirely.

ii.
```python
for anm, date, probes, data_dir in SESSION_META:
    sess = load_session(anm, date, probes, data_dir)
    result = process_session(sess)
    if result is None:
        continue
```

```python
if good_units.sum() < MIN_UNITS:
    print(f"  Skipping {sess['anm']}_{sess['date']}: only {good_units.sum()} units after FR filter")
    return None
```

iii. The agent identified sessions from the loading scripts and added a minimum-units filter (10 units) as a quality control.

## 1-d. How are the data split into trials?

i. Each session has `Ntrials` trials as defined by `bp.Ntrials`. Per-trial fields (`hit`, `miss`, `R`, `L`, `autowater`, `early`, `stim.enable`, `goCue`) are read as flat arrays of length `ntrials`. Spike times carry trial indices (`clu.trial`) for assignment to trials.

ii.
```python
ntrials = int(bp['Ntrials'][0, 0])
hit = np.array(bp['hit']).flatten().astype(bool)
...
for ti in range(ntrials):
    spk_mask = spike_trials_raw == (ti + 1)
```

iii. The agent used the Bpod trial structure directly, where each trial is defined by its index and associated metadata fields.

## 1-e. How are trials filtered based on quality controls?

i. Two filters are applied: early-lick trials (`bp.early`) and photostimulation trials (`bp.stim.enable`) are excluded. No filter is applied for trials that extend past the end of the neural recording.

ii.
```python
trial_mask = ~sess['stim_enable'] & ~sess['early']
valid_trials = np.where(trial_mask)[0]
```

iii. The agent followed the paper's convention of excluding early-lick and photostim trials. The agent did not implement a check for trials extending past the end of the recording.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` spike-sorted clusters, specifically `clu.trial` (which trial each spike belongs to, 1-based) and `clu.trialtm` (spike time relative to trial start). Also `bp.ev.goCue` for alignment.

ii.
```python
spike_trials_raw = sess['spike_trials'][ui]
spike_trialtm_raw = sess['spike_trialtm'][ui]
aligned_times = spike_trialtm_raw[spk_mask] - goCue[ti]
```

iii. The agent identified `trial` and `trialtm` fields from the cluster data as the primary spike timing information, aligned to go cue.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to go cue, binned into 10 ms bins (DT = 1/100), converted to firing rates (Hz), then smoothed with a causal half-Gaussian kernel of window size 15 bins. The causal kernel is `gausswin(15)` with the first half zeroed out. Boundary condition is 'reflect' (prepend N reflected samples before convolution).

ii.
```python
DT = 1.0 / 100  # 10 ms bins
SMOOTH_WINDOW = 15  # bins for causal Gaussian kernel

def causal_gaussian_kernel(N):
    alpha = 2.5
    n = np.arange(N)
    w = np.exp(-0.5 * (alpha * (n - (N - 1) / 2) / ((N - 1) / 2)) ** 2)
    w[:N // 2] = 0
    w /= w.sum()
    return w

def smooth_data(x, N, bc_type='reflect'):
    kern = causal_gaussian_kernel(N)
    if bc_type == 'reflect':
        x_padded = np.concatenate([x[:N], x], axis=0)
    ...
    out[:, j] = np.convolve(x_padded[:, j], kern, mode='same')
```

iii. The agent read `WorkingWithDataObjs.m` which specifies `params.dt = 1/100` and `params.smooth = 15`, and `mySmooth.m` which uses `gausswin(N)` with causal zeroing. The agent interpreted `params.dt = 1/100` as 10 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, clusters with quality labels in `{'garbage', 'gabrga', 'noisy', 'real?'}` are excluded (case-insensitive). Then units with mean firing rate ≤ 1 Hz are removed. Sessions with fewer than 10 remaining units are dropped entirely.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR_THRESH = 1.0
MIN_UNITS = 10

if q_str.lower() in {q.lower() for q in EXCLUDED_QUALITIES}:
    continue
...
mean_frs = np.mean(trialdat[:, :, valid_trials], axis=(0, 2))
good_units = mean_frs > LOW_FR_THRESH
if good_units.sum() < MIN_UNITS:
    return None
```

iii. The agent read `findClusters.m` which defines the quality filter as 'all' excluding `{'garbage', 'gabrga', 'noisy', 'real?'}`, and `removeLowFRClusters.m` for the 1 Hz threshold. The MIN_UNITS=10 filter was added by the agent as additional quality control.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times (`trialtm`) are aligned to the go cue by subtracting `goCue[trial]` from each spike's trial-relative time. This produces spike times in seconds from go cue onset.

ii.
```python
aligned_times = spike_trialtm_raw[spk_mask] - goCue[ti]
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The agent followed the reference `alignSpikes.m` which aligns to `params.alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 10 ms (`DT = 1/100`), producing 500 bins across the -2.5 to 2.5 s window. No rebinning is applied; spikes are directly histogrammed into 10 ms bins.

ii.
```python
DT = 1.0 / 100  # 10 ms bins
edges = np.arange(TMIN, TMAX + DT, DT)
time_centers = edges[:-1] + DT / 2
```

iii. The agent interpreted `params.dt = 1/100` from `WorkingWithDataObjs.m` as 10 ms (0.01 s) bins.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This variable is constructed from the time bin centers, not from raw data. It represents the center of each time bin relative to go cue onset.

ii.
```python
taxis = time_centers  # edges[:-1] + DT/2
input_trials.append(taxis.reshape(1, -1).astype(np.float32))
```

iii. The time axis is defined by the binning grid parameters.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing beyond computing bin centers from the bin edges.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_centers = edges[:-1] + DT / 2
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the same time axis used for neural binning. The same `edges` array defines both the spike histogram bins and the input time values.

ii.
```python
counts, _ = np.histogram(aligned_times, bins=edges)
...
input_trials.append(taxis.reshape(1, -1).astype(np.float32))
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Three per-trial fields: `bp.hit`, `bp.miss`, and `bp.R` (right trial indicator). Lick direction is inferred from the combination of trial outcome and instructed direction.

ii.
```python
if sess['hit'][ti]:
    lick_dir = 1 if sess['R'][ti] else 0
elif sess['miss'][ti]:
    lick_dir = 0 if sess['R'][ti] else 1  # opposite of cue
else:
    lick_dir = 2  # ignore/no response
```

iii. The agent reasoned that `R` and `L` indicate the cue direction, not the animal's actual lick. Hit means the animal licked the correct side, miss means the opposite.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Hit trials: lick matches cue direction (R→right=1, L→left=0). Miss trials: lick is opposite of cue (R→left=0, L→right=1). Ignore trials: no lick (2).

ii. Same as 4-a code.

iii. The agent initially missed the "none" category for ignore trials, then fixed it after noticing lick_direction only had values {0, 1}.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater` — a per-trial flag indicating water-cued (WC) context.

ii.
```python
context = 0 if sess['autowater'][ti] else 1
```

iii. The agent identified autowater as the WC indicator.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct mapping: autowater=True → WC (0), autowater=False → DR (1).

ii. Same as 5-a code.

iii. Follows the prompt's WC/DR specification.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit` and `bp.miss` per-trial flags.

ii.
```python
outcome = 1 if sess['hit'][ti] else (0 if sess['miss'][ti] else 2)
```

iii. The agent used the same hit/miss flags as for lick direction.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit → correct (1), miss → incorrect (0), neither → ignore (2).

ii. Same as 6-a code.

iii. Follows the prompt's coding scheme.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. DeepLabCut tracking from `obj.traj`, specifically the bottom camera only. The `top_tongue` feature's x, y coordinates and confidence (likelihood) values are used. Frame times from `traj.frameTimes` and the video offset from `sglx.bitcode` are also used.

ii.
```python
tongue_idx = bottom_feat_names.index('top_tongue')
...
trial_tongue_xy.append(ts[tongue_idx, :2, :].T)
trial_tongue_conf.append(ts[tongue_idx, 2, :])
```

iii. The agent identified `top_tongue` from the bottom camera as the tongue tracking feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Five steps: (1) Compute velocity as `sqrt(dx^2 + dy^2) * 400` using frame-to-frame differences at 400 Hz. (2) Frames with DLC confidence < 0.9 are marked not visible. (3) NaN values are NOT filled for tongue (per methods: "except for the tongue"). (4) Velocity is linearly interpolated from frame times to the 10 ms bin centers. (5) Split at per-session 50th percentile of visible values; non-visible bins get class 2.

ii.
```python
def compute_velocity(xy_coords, confidence=None, fill_missing=True):
    dx = np.diff(xy[:, 0])
    dy = np.diff(xy[:, 1])
    vel = np.sqrt(dx**2 + dy**2) * VIDEO_FPS
    vel = np.concatenate([[vel[0]], vel])
    vel[~visible] = np.nan
    return vel, visible

# For tongue: fill_missing=False
vel, vis = compute_velocity(tongue_xy, confidence=tongue_conf, fill_missing=False)
f_interp = interp1d(ft_aligned, vel, kind='linear', bounds_error=False, fill_value=np.nan)
v = f_interp(taxis)
```

iii. The agent initially filled all missing values, then discovered this made all tongue values above threshold. After examining DLC confidence, the agent learned that tongue has mostly low confidence (~84% below 0.5) since it's only visible during licking, and corrected to not fill tongue missing values per the methods text.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile of visible (non-NaN, confidence >= 0.9) tongue velocity values. Below threshold → 0, at or above → 1, not visible → 2.

ii.
```python
t_vals = tongue_vel_aligned[:, valid_trials][tongue_visible[:, valid_trials]]
t_vals = t_vals[~np.isnan(t_vals)]
tongue_thresh = np.percentile(t_vals, 50) if len(t_vals) > 0 else 0

tongue_disc = np.full(n_timebins, 2, dtype=np.int64)
vis = tongue_visible[:, ti] & ~np.isnan(tongue_vel_aligned[:, ti])
tongue_disc[vis & (tongue_vel_aligned[:, ti] < tongue_thresh)] = 0
tongue_disc[vis & (tongue_vel_aligned[:, ti] >= tongue_thresh)] = 1
```

iii. Follows the instruction's 50th percentile discretization scheme.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by subtracting the video offset and go cue time, then tongue velocity is linearly interpolated from the corrected frame times onto the neural time bin centers using `scipy.interpolate.interp1d`.

ii.
```python
ft_aligned = ft - vidshift - goCue[ti]
f_interp = interp1d(ft_aligned, vel, kind='linear', bounds_error=False, fill_value=np.nan)
v = f_interp(taxis)
```

iii. The video offset is computed as `mode(bitcode.bitstart)/sglx.fs - mode(ev.bitStart)`, matching `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DeepLabCut tracking from the bottom camera only, specifically `top_paw` feature's x, y coordinates and confidence.

ii.
```python
paw_idx = bottom_feat_names.index('top_paw')
trial_paw_xy.append(ts[paw_idx, :2, :].T)
trial_paw_conf.append(ts[paw_idx, 2, :])
```

iii. The agent selected `top_paw` from the bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same velocity computation as tongue (`sqrt(dx^2+dy^2)*400Hz`), but with missing values filled using nearest-neighbor interpolation (per the methods: fill missing for all features except tongue). Velocity is then linearly interpolated to bin centers, and remaining NaN values are filled with `fill_nearest`. Split at per-session 50th percentile.

ii.
```python
vel, vis = compute_velocity(paw_xy, confidence=paw_conf, fill_missing=True)
f_interp = interp1d(ft_aligned, vel, kind='linear', bounds_error=False, fill_value=np.nan)
paw_vel_aligned[:, ti] = f_interp(taxis)
...
fill_nearest(paw_vel_aligned)
```

iii. The agent followed the methods text that missing values should be filled for all features except the tongue.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Per-session 50th percentile of visible paw velocity values. Below threshold → 0, at or above → 1, not visible → 2.

ii.
```python
p_vals = paw_vel_aligned[:, valid_trials][paw_visible[:, valid_trials]]
p_vals = p_vals[~np.isnan(p_vals)]
paw_thresh = np.percentile(p_vals, 50) if len(p_vals) > 0 else 0

paw_disc = np.full(n_timebins, 2, dtype=np.int64)
vis = paw_visible[:, ti] & ~np.isnan(paw_vel_aligned[:, ti])
paw_disc[vis & (paw_vel_aligned[:, ti] < paw_thresh)] = 0
paw_disc[vis & (paw_vel_aligned[:, ti] >= paw_thresh)] = 1
```

iii. Same discretization scheme as tongue.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times corrected by video offset and go cue, then linearly interpolated to bin centers.

ii.
```python
ft_aligned = ft - vidshift - goCue[ti]
f_interp = interp1d(ft_aligned, vel, kind='linear', bounds_error=False, fill_value=np.nan)
paw_vel_aligned[:, ti] = f_interp(taxis)
```

iii. Same video offset and alignment as tongue.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Separate `motionEnergy_<anm>_<date>.mat` files. The `me.data` field contains per-trial motion energy traces (one value per camera frame).

ii.
```python
me_mat = sio.loadmat(me_fpath, squeeze_me=True)
me = me_mat['me']
me_data_raw = me['data'].item()
sess['me_data'] = [me_data_raw[i] for i in range(len(me_data_raw))]
```

iii. The agent loaded motion energy from the separate files, handling both scipy and HDF5 formats.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values (already one value per frame) are linearly interpolated from frame times to bin centers, then NaN values are filled with nearest-neighbor. Split at per-session 50th percentile.

ii.
```python
f_interp = interp1d(ft_aligned, me_trial, kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, ti] = f_interp(taxis)
...
fill_nearest(me_aligned)
```

iii. The agent treated motion energy as a pre-computed per-frame quantity needing only temporal alignment.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile of non-NaN motion energy values. Below → 0, at or above → 1, NaN/missing → 2 ("no video").

ii.
```python
m_vals = me_aligned[:, valid_trials]
m_vals = m_vals[~np.isnan(m_vals)]
me_thresh = np.percentile(m_vals, 50)

me_disc = np.full(n_timebins, 2, dtype=np.int64)
valid_me = ~np.isnan(me_aligned[:, ti])
me_disc[valid_me & (me_aligned[:, ti] < me_thresh)] = 0
me_disc[valid_me & (me_aligned[:, ti] >= me_thresh)] = 1
```

iii. Follows instruction's discretization scheme.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the same frame times as the bottom camera tracking. Frame times are corrected by video offset and go cue, then motion energy is linearly interpolated to bin centers.

ii.
```python
ft_aligned = ft - vidshift - goCue[ti]
f_interp = interp1d(ft_aligned, me_trial, kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, ti] = f_interp(taxis)
```

iii. The agent assumed motion energy frame times match the bottom camera's frame times.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) DLC confidence < 0.9 marks frames as not visible for tongue and paw. (2) For tongue, NaN values are NOT filled (per methods). (3) For paw and motion energy, NaN values ARE filled with nearest-neighbor via `fill_nearest`. (4) Empty frame time arrays cause the trial's video data to remain NaN. (5) Trials with all-NaN video data get class 2 ("not visible"/"no video"). (6) Sessions loading errors are caught and skipped with a warning.

ii.
```python
def fill_nearest(arr):
    nans = np.isnan(arr)
    valid = np.where(~nans)[0]
    for idx in np.where(nans)[0]:
        nearest = valid[np.argmin(np.abs(valid - idx))]
        arr[idx] = arr[nearest]
```

```python
vel[~visible] = np.nan
```

iii. The agent followed the methods text that says "Missing values were filled in with the nearest available value for all features, except for the tongue."

## 11-a. What are the most time-consuming steps of the code?

i. File I/O dominates: loading the HDF5/v5 MATLAB files is slow. Additionally, the triple nested loop (sessions × units × trials) for spike binning is computationally expensive since it histograms spikes one unit at a time, one trial at a time.

ii.
```python
for ui in range(n_units):
    for ti in range(ntrials):
        spk_mask = spike_trials_raw == (ti + 1)
        counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The agent noted that loading dominates runtime.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates over every unit and every trial individually, creating a boolean mask and histogram for each. This could be vectorized using `np.histogram2d` (as the reference does) to bin all trials for a unit in one call. The velocity interpolation loop over trials could also be partially vectorized.

ii.
```python
for ui in range(n_units):
    for ti in range(ntrials):
        spk_mask = spike_trials_raw == (ti + 1)
        aligned_times = spike_trialtm_raw[spk_mask] - goCue[ti]
        counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The inner trial loop is the most obvious candidate for vectorization.

## 11-c. What processing does the code repeat multiple times?

i. The smoothing is applied per-unit per-trial, which repeats the kernel construction and convolution setup. However, the kernel itself is constructed fresh each time `smooth_data` is called. More significantly, the frame time alignment (`ft - vidshift - goCue[ti]`) is computed separately for tongue, paw, and motion energy even though they share the same frame times (all from bottom camera).

ii.
```python
# Frame time correction repeated for each feature:
ft_aligned = ft - vidshift - goCue[ti]
# Then interpolation done separately for tongue, paw, ME
```

iii. The agent did not explicitly optimize for repeated computations.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes spike binning and smoothing for ALL trials (including filtered ones) before selecting valid trials. This means early-lick and stim trials are fully processed before being discarded. The `fill_nearest` on paw and ME data also operates on all trials including filtered ones.

ii.
```python
trialdat = np.zeros((n_timebins, n_units, ntrials))  # ALL trials
for ui in range(n_units):
    for ti in range(ntrials):  # processes every trial
        ...
# Only later:
valid_trials = np.where(trial_mask)[0]
```

iii. The agent prioritized correctness over efficiency, processing all trials then filtering.
