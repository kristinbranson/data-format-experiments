# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes 44 sessions in two lists (`EPHYS_SESSIONS` and `RD_SESSIONS`), each specifying animal name, date, and probe numbers. Each session's `.mat` file is loaded with either `h5py` (for v7.3 HDF5) or `scipy.io.loadmat` (for v5), tried in that order. Motion energy is loaded separately from `motionEnergy_<anm>_<date>.mat` files.

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

def load_mat_file(fpath):
    try:
        f = h5py.File(fpath, 'r')
        return f, 'h5py'
    except:
        data = sio.loadmat(fpath, squeeze_me=False)
        return data, 'scipy'
```

iii. The AI identified the session lists from the authors' loading scripts and correctly included all 44 sessions (25 fixed-delay, 19 randomized-delay). The CONVERSION_NOTES.md documents the excluded files and why.

## 1-b. How are the data split into subjects?

i. The animal name is the first element of each session tuple (e.g., `'EKH1'`). Subjects are collected into a list in the order they first appear, and `subject_idx` maps each session to its index.

ii.
```python
if anm not in subjects:
    subjects.append(anm)
subject_idx.append(subjects.index(anm))
```

iii. The AI uses the animal name directly from the session definition. This produces 14 unique subjects.

## 1-c. How are the data split into sessions?

i. Each tuple in `EPHYS_SESSIONS` or `RD_SESSIONS` defines one session. The two lists are concatenated, with each entry pointing to its respective data directory. Each session becomes one element of the output lists.

ii.
```python
all_sessions = [(a, d, p, EPHYS_DATA_DIR) for a, d, p in EPHYS_SESSIONS] + \
               [(a, d, p, RD_DATA_DIR) for a, d, p in RD_SESSIONS]
```

iii. The data directories are hard-coded for each task type.

## 1-d. How are the data split into trials?

i. The number of trials per session is read from `bp.Ntrials`. All arrays (hit, miss, R, L, etc.) are read with that length. Trials are indexed 0 to n_trials-1.

ii.
```python
info = {
    'n_trials': int(bp['Ntrials'][()].flatten()[0]),
    'hit': bp['hit'][()].flatten().astype(bool),
    ...
}
```

iii. The AI reads the trial count from the Bpod structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI does NOT filter out early-lick or photostimulation trials. Instead, it overrides the outcome and lick direction labels for early-lick trials (setting them to 'ignore' and 'none' respectively). Photostimulation trials are included without any special handling. No recording-length filter is applied, resulting in trials with all-zero neural data in some sessions.

ii.
```python
# Early lick trials: override to 'ignore' regardless of hit/miss
lick_direction[early] = 2
outcome[early] = 2
```

iii. The AI's CONVERSION_NOTES.md under Step 5 states "No trial filtering by condition - all trials used, with conditions as output labels." The Step 10 notes document the discovery and override of early-lick labels but do not remove the trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` spike-sorted clusters, specifically `trial` (which trial each spike belongs to), `trialtm` (spike time relative to trial start), and `quality` (curation label). Also `bp.ev.goCue` for alignment.

ii.
```python
trialtm = f[clu['trialtm'][i, 0]][()].flatten()
trial = f[clu['trial'][i, 0]][()].flatten().astype(int)
```

iii. Same raw variables as the reference.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to go cue, binned into 10ms bins (DT=1/100), converted to firing rates (Hz), then smoothed with a causal Gaussian kernel (window=15 samples, reflect boundary). The causal kernel zeros out the first half of a Gaussian window, so only past/current time points contribute.

ii.
```python
DT = 1.0 / 100  # 10ms bins
SMOOTH_WINDOW = 15
...
aligned_times = clu['trialtm'][spk_mask] - goCue[trial_num - 1]
N, _ = np.histogram(aligned_times, bins=edges)
fr = N.astype(np.float64) / DT
trialdat[:, ci, trial_num - 1] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

The causal smoothing function:
```python
def causal_gaussian_smooth(x, N, bctype='reflect'):
    kern = np.exp(-0.5 * (alpha * (n - center) / center) ** 2)
    kern[:int(N // 2)] = 0  # zero out first half for causal
    kern = kern / kern.sum()
```

iii. The CONVERSION_NOTES.md states the bin size was chosen as 10ms from `WorkingWithDataObjs.m` rather than 5ms from `getDefaultParams.m`. The smoothing is described as matching `mySmooth.m` (causal Gaussian, reflect BC).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Clusters with quality labels in `{garbage, noisy, gabrga, real?}` are excluded, AND clusters with empty quality labels are also excluded. (2) Units with mean firing rate <= 1 Hz are removed.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'noisy', 'gabrga', 'real?'}
...
q_str = read_h5_string(f, quality_ds[i, 0]).lower()
if q_str in EXCLUDED_QUALITIES or q_str == '' or '\x00' in q_str:
    continue
...
mean_fr = trialdat.mean(axis=(0, 2))
keep_mask = mean_fr > LOW_FR_THRESHOLD
```

iii. The AI notes this matches `findClusters.m` which excludes garbage, noisy, gabrga, real?. However, 'poor' quality units are not in the exclusion set (the reference also excludes 'poor'), and empty quality labels are excluded (the reference keeps them).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times (`trialtm`) are aligned to the go cue by subtracting `goCue[trial_num - 1]` from each spike time. This is done before binning.

ii.
```python
aligned_times = clu['trialtm'][spk_mask] - goCue[trial_num - 1]
N, _ = np.histogram(aligned_times, bins=edges)
```

iii. Matches the reference approach of `trialtm - goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10ms bins (DT = 1/100), producing 500 time bins over the [-2.5, 2.5]s window. No rebinning is applied.

ii.
```python
DT = 1.0 / 100  # 10ms bins
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
n_time = len(time_axis)  # 500
```

iii. The AI chose 10ms based on `WorkingWithDataObjs.m` tutorial code, noting both 5ms and 10ms are used in the codebase. The reference uses 5ms (matching `getDefaultParams.m`'s `dt = 1/200`).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Derived from the time axis computed from the bin edges, not from any raw data variable. It is the center of each time bin.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
```

iii. The time axis is constructed from the binning parameters.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing beyond computing bin centers from the edge array.

ii.
```python
time_input = time_axis.astype(np.float32).reshape(1, -1)
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis IS the neural binning grid. Both neural data and time input share the same bin edges, so bin k represents the same time interval.

ii.
```python
N, _ = np.histogram(aligned_times, bins=edges)
...
time_input = time_axis.astype(np.float32).reshape(1, -1)
```

iii. Same bin structure ensures alignment.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. `bp.hit`, `bp.miss`, `bp.R`, `bp.L`, and `bp.early`.

ii.
```python
hit = trial_info['hit']
miss = trial_info['miss']
R = trial_info['R']
L = trial_info['L']
early = trial_info['early']
```

iii. These are read from the Bpod trial structure.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit on a right trial means right lick; a hit on a left trial means left lick. A miss on a right trial means left lick (wrong); a miss on a left trial means right lick (wrong). All other trials (including early-lick trials, which are overridden) get 'none'. Codes: left=0, right=1, none=2.

ii.
```python
lick_direction = np.full(n_trials, 2, dtype=int)  # 2=none
lick_direction[hit & R] = 1  # right
lick_direction[hit & L] = 0  # left
lick_direction[miss & R] = 0  # licked left (wrong on right trial)
lick_direction[miss & L] = 1  # licked right (wrong on left trial)
lick_direction[early] = 2  # override early lick
```

iii. The derivation logic is the same as the reference. Early lick trials are overridden to 'none' rather than being excluded.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater`, which marks water-cued (WC) trials.

ii.
```python
autowater = trial_info['autowater']
```

iii. Read directly from the trial table.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater trials become WC (code 1), others become DR (code 0). Output values: `['DR', 'WC']`.

ii.
```python
context = autowater.astype(int)  # 0=DR, 1=WC
```

iii. The coding is inverted compared to the reference (which uses WC=0, DR=1), but the output_values labels match the coding, so the semantic meaning is preserved.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit`, `bp.miss`, and `bp.early`.

ii.
```python
hit = trial_info['hit']
miss = trial_info['miss']
early = trial_info['early']
```

iii. Same source variables as reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit trials are 'correct' (1), miss trials are 'incorrect' (0), all others are 'ignore' (2). Early-lick trials are overridden to 'ignore' regardless of their hit/miss flags.

ii.
```python
outcome = np.full(n_trials, 2, dtype=int)  # 2=ignore
outcome[hit] = 1  # correct
outcome[miss] = 0  # incorrect
outcome[early] = 2  # override
```

iii. The override for early-lick trials was added during Critical Review 1. The reference instead removes these trials entirely.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. DeepLabCut tracking from `obj.traj`, specifically the 'tongue' feature from the side camera (view index 0). The x, y positions and confidence values are extracted.

ii.
```python
if 'tongue' in feat_side:
    ti = feat_side.index('tongue')
    tx, ty, tc = ts_side[ti, 0, :], ts_side[ti, 1, :], ts_side[ti, 2, :]
```

iii. Only the side camera is used. The reference uses both side ('tongue') and bottom ('top_tongue') cameras.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. (1) Speed is computed as `sqrt(dx^2 + dy^2) / dt` using np.diff on raw x,y positions (no smoothing before differentiation). (2) Frames with DLC confidence < 0.5 are set to NaN. (3) The speed is interpolated to the time axis using np.interp. (4) Discretized at the per-session 50th percentile, with a 'not visible' class for NaN bins.

ii.
```python
DLC_CONFIDENCE_THRESHOLD = 0.5
...
tspeed = compute_velocity(tx, ty, ft_side)
tspeed[~tongue_visible] = np.nan
tv_interp = np.interp(time_axis, ft_side_aligned, tspeed)
tv_vis = np.interp(time_axis, ft_side_aligned, tongue_visible.astype(float)) >= 0.5
tv_interp[~tv_vis] = np.nan
...
tongue_thresh = np.nanpercentile(tongue_vel_all[tongue_valid], 50)
tongue_disc[tongue_valid & (tongue_vel_all < tongue_thresh)] = 0
tongue_disc[tongue_valid & (tongue_vel_all >= tongue_thresh)] = 1
```

iii. Major differences from reference: (a) no smoothing of x,y before differentiation, (b) np.diff instead of np.gradient, (c) confidence threshold 0.5 vs 0.9, (d) only side camera vs both cameras, (e) interpolation vs bin-averaging, (f) velocity computed on all frames then masked, vs only on valid contiguous runs.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile of all valid (non-NaN) tongue velocity values across all trials. Values below threshold = 0, at or above = 1, NaN (not visible) = 2.

ii.
```python
tongue_thresh = np.nanpercentile(tongue_vel_all[tongue_valid], 50)
tongue_disc = np.full(tongue_vel_all.shape, 2, dtype=int)
tongue_disc[tongue_valid & (tongue_vel_all < tongue_thresh)] = 0
tongue_disc[tongue_valid & (tongue_vel_all >= tongue_thresh)] = 1
```

iii. Matches the instruction's specification for per-session 50th percentile thresholding.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The video offset is computed from bitcode timestamps (mode of sglx.bitcode.bitstart/fs minus mode of bp.ev.bitStart). Frame times are corrected by subtracting the offset and the trial's go cue time. The corrected frame times are then used with np.interp to interpolate onto the neural time axis.

ii.
```python
info['vidshift'] = sp_stats.mode(bitstart_sglx, keepdims=False).mode / fs - sp_stats.mode(bitStart_bp, keepdims=False).mode
...
ft_side_aligned = ft_side - vidshift - goCue[t]
tv_interp = np.interp(time_axis, ft_side_aligned, tspeed)
```

iii. The video offset computation matches `findVideoOffset.m`. The alignment approach differs from the reference which bins frames into the bin grid by averaging.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DLC tracking of 'top_paw' from the bottom camera (view index 1).

ii.
```python
if 'top_paw' in feat_bot:
    pi = feat_bot.index('top_paw')
    px, py, pc = ts_bot[pi, 0, :], ts_bot[pi, 1, :], ts_bot[pi, 2, :]
```

iii. Same feature and camera as the reference.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as tongue: raw np.diff velocity, DLC confidence threshold of 0.5, np.interp to time axis, per-session 50th percentile discretization.

ii.
```python
pspeed = compute_velocity(px, py, ft_bot)
pspeed[~paw_visible] = np.nan
pv_interp = np.interp(time_axis, ft_bot_aligned, pspeed)
```

iii. Same differences from reference as tongue velocity (no smoothing, np.diff, confidence 0.5, interpolation).

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Per-session 50th percentile, same scheme as tongue. Below = 0, at or above = 1, not visible = 2.

ii.
```python
paw_thresh = np.nanpercentile(paw_vel_all[paw_valid], 50)
paw_disc[paw_valid & (paw_vel_all < paw_thresh)] = 0
paw_disc[paw_valid & (paw_vel_all >= paw_thresh)] = 1
```

iii. Matches instruction specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: video offset correction, go cue subtraction, np.interp to time axis.

ii.
```python
ft_bot_aligned = ft_bot - vidshift - goCue[t]
pv_interp = np.interp(time_axis, ft_bot_aligned, pspeed)
```

iii. Same approach as tongue alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Separate `motionEnergy_<anm>_<date>.mat` files containing per-trial motion energy traces (one value per camera frame).

ii.
```python
def load_motion_energy(anm, date, data_dir):
    me_data = sio.loadmat(me_path, squeeze_me=False)
    me_raw = me_data['me']
    # handles struct, nested struct, and cell array formats
```

iii. The AI handles the three different file layouts (bare cell array, struct with data, nested struct).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The motion energy trace is interpolated to the neural time axis using np.interp (using side camera frame times), then discretized at the per-session 50th percentile.

ii.
```python
if len(me_trial) == len(ft_side):
    me_interp = np.interp(time_axis, ft_side_aligned, me_trial)
...
me_thresh_disc = np.nanpercentile(me_all[me_valid], 50)
me_disc[me_valid & (me_all < me_thresh_disc)] = 0
me_disc[me_valid & (me_all >= me_thresh_disc)] = 1
```

iii. Reference uses bin-averaging instead of interpolation.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile. Below = 0, at or above = 1, no video = 2.

ii.
```python
me_thresh_disc = np.nanpercentile(me_all[me_valid], 50)
me_disc = np.full(me_all.shape, 2, dtype=int)
me_disc[me_valid & (me_all < me_thresh_disc)] = 0
me_disc[me_valid & (me_all >= me_thresh_disc)] = 1
```

iii. Matches instruction specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same video offset and go cue correction as tongue/paw. The side camera frame times are used since motion energy has one value per side camera frame. np.interp interpolates to the time axis.

ii.
```python
if len(me_trial) == len(ft_side):
    me_interp = np.interp(time_axis, ft_side_aligned, me_trial)
```

iii. Same alignment approach as other camera streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Exceptions during per-trial DLC/ME processing are silently caught, leaving NaN arrays that become 'not visible' class 2. If motion energy length doesn't match frame times, the AI tries to construct synthetic frame times using linspace. Trials with all-zero neural data (from recording ending early) are kept.

ii.
```python
except Exception as e:
    pass  # Leave as NaN
...
if len(me_trial) == len(ft_side):
    me_interp = np.interp(time_axis, ft_side_aligned, me_trial)
elif len(me_trial) > 0:
    me_ft = np.linspace(ft_side[0], ft_side[-1], len(me_trial))
```

iii. The silent exception handling is a broad catch-all. The reference explicitly handles missing frame times and mismatched frame counts. The AI does not filter out trials past the end of the recording, resulting in ~60 trials with all-zero neural data.

## 11-a. What are the most time-consuming steps of the code?

i. The per-cluster, per-trial spike binning loop is the most expensive, as it iterates over every cluster and every trial individually (nested loop). File loading is also expensive.

ii.
```python
for ci, clu in enumerate(clusters):
    for trial_num in range(1, n_trials + 1):
        spk_mask = clu['trial'] == trial_num
        ...
        N, _ = np.histogram(aligned_times, bins=edges)
```

iii. The double loop over clusters and trials is much slower than the reference's vectorized histogram2d approach.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop (cluster x trial) could be vectorized using `np.histogram2d` as the reference does. The per-trial DLC processing loop could partially be vectorized. The per-trial output construction loop is also unnecessary.

ii.
```python
# This nested loop:
for ci, clu in enumerate(clusters):
    for trial_num in range(1, n_trials + 1):
        ...
# could be replaced with histogram2d as in the reference
```

iii. The reference bins all spikes at once with `np.histogram2d(spike_trial, spike_time, bins=[trial_edges, BIN_EDGES])`.

## 11-c. What processing does the code repeat multiple times?

i. The per-trial DLC trajectory extraction is done for every trial in a single loop that handles side camera, bottom camera, and motion energy together, so there is no redundant loading. However, the h5py file handle is kept open throughout processing and only closed after all trials are done. The `_traj` data is re-read for each trial (side and bottom views) without caching.

ii.
```python
for t in range(n_trials):
    ft_side, ts_side, feat_side = get_traj_trial_h5(file_data, obj, t, 0)
    ft_bot, ts_bot, feat_bot = get_traj_trial_h5(file_data, obj, t, 1)
```

iii. Feature name lookup is repeated every trial but this is minor.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes velocity for ALL frames (including low-confidence ones) before masking them as NaN. It also loads `bp.ev.sample` and `bp.ev.delay` which are never used. The `bp.L` field is loaded but could be derived from `~R`. The `show_processing` flag is accepted but the plotting code is not implemented.

ii.
```python
'sample': ev['sample'][()].flatten(),
'delay': ev['delay'][()].flatten(),
...
tspeed = compute_velocity(tx, ty, ft_side)  # computed for all frames
tspeed[~tongue_visible] = np.nan  # then masked
```

iii. The velocity on low-confidence frames is wasted computation.
