# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB file `data_structure_<anm>_<date>.mat` in either `Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior` directories. The 44 sessions are hard-coded in `SESSION_META` with animal, date, dataset directory, and probe list. Files are loaded using either h5py (HDF5/v7.3) or scipy.io (v5), determined by trying h5py first. Motion energy is loaded from separate `motionEnergy_*.mat` files.

ii.
```python
SESSION_META = [
    ('EKH1', '2021-08-07', 'Ephys_Behavior', [2]),
    ...
    ('JEB24', '2023-11-03', 'RandomizedDelay_Ephys_Behavior', [1]),
]

def load_mat_file(fpath):
    try:
        f = h5py.File(fpath, 'r')
        return f, 'h5'
    except:
        data = sio.loadmat(fpath, squeeze_me=False)
        return data, 'v5'
```

iii. The session list was derived from the authors' `load<ANM>_ALMVideo.m` scripts. The AI documented this in CONVERSION_NOTES.md Step 1.

## 1-b. How are the data split into subjects?

i. The animal name is the first element of each `SESSION_META` tuple (e.g., `'EKH1'`). At assembly, subjects are the sorted set of unique animal names, and `subject_idx` maps each session to its subject.

ii.
```python
all_animals = sorted(set(r['anm'] for r in all_results))
subjects = all_animals
subject_idx = np.array([subjects.index(r['anm']) for r in all_results])
```

iii. The animal name is explicitly stored in the session metadata tuple rather than parsed from the filename.

## 1-c. How are the data split into sessions?

i. Each entry in `SESSION_META` is one session, identified by `(anm, date, dataset_dir, alm_probes)`. Each is processed by `process_session()` and becomes one element in the output lists. Fixed-delay and randomized-delay sessions are handled uniformly. Result: 44 sessions (25 fixed-delay, 19 randomized-delay).

ii.
```python
for i, (anm, date, dataset_dir, alm_probes) in enumerate(sessions_to_process):
    result = process_session(anm, date, dataset_dir, alm_probes, ...)
    if result is not None:
        all_results.append(result)
```

iii. The AI documented 44 total sessions matching the paper.

## 1-d. How are the data split into trials?

i. Trials correspond to the entries of the behavioral parameter arrays (bp fields). The number of trials is read from `bp.Ntrials`. Each per-trial field (hit, miss, early, etc.) is read as a flat array of that length. Spike data carries trial numbers (1-indexed), and spikes are binned per trial.

ii.
```python
ntrials = int(f['obj/bp/Ntrials'][0, 0])
hit = f['obj/bp/hit'][:].flatten().astype(bool)
...
for tr_idx in range(ntrials):
    trial_num = tr_idx + 1
    mask = spk_trials == trial_num
```

iii. The Bpod table defines trials directly.

## 1-e. How are trials filtered based on quality controls?

i. Two filters: early-lick trials (`bp.early`) and photostimulation trials (`bp.stim.enable`) are excluded. No filtering based on recording length (trials past the end of the recording are not explicitly detected or removed).

ii.
```python
valid_trials_mask = ~early & ~stim_enable
valid_trial_indices = np.where(valid_trials_mask)[0]
```

iii. The AI documented "Exclude early lick and stimulation trials" following the paper and reference code. The AI did not implement a check for trials running past the end of the recording.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` spike-sorted clusters. Each cluster provides `trial` (1-indexed trial number for each spike), `trialtm` (spike time relative to trial start), and `quality` (curation label). Go cue times from `bp.ev.goCue` are used for alignment.

ii.
```python
trialtm = f[trialtm_ref][:].flatten()
trial = f[trial_ref][:].flatten().astype(int)  # 1-indexed
aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]
```

iii. Documented in CONVERSION_NOTES.md Step 1: "alignSpikes → Align spike times to event (goCue)".

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to go cue (trialtm - goCue), then binned into 10ms bins spanning -2.5 to 2.5s (500 bins). Counts are divided by DT to get firing rates in Hz. Rates are smoothed with a **causal** Gaussian kernel of 15 samples (the first half is zeroed). The kernel uses reflect boundary conditions.

ii.
```python
DT = 1.0 / 100  # 10ms bins
SMOOTH_WIN = 15

def causal_gaussian_kernel(n):
    alpha = 2.5
    half = (n_pts - 1) / 2
    t = np.arange(n_pts) - half
    kern = np.exp(-0.5 * (alpha * t / half) ** 2)
    kern[:n_pts // 2] = 0  # Make causal
    kern = kern / kern.sum()
    return kern

counts, _ = np.histogram(spk_t, bins=time_edges)
fr = counts.astype(np.float32) / DT
trialdat[:, i, tr_idx] = smooth_with_bc(fr, kernel, BOUNDARY_CONDITION)
```

iii. The AI justified 10ms bins based on `WorkingWithDataObjs.m` tutorial and the causal Gaussian based on the reference `mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. (1) Clusters whose quality label matches `{garbage, gabrga, noisy, real?}` are excluded. The match is **case-sensitive**. The label `poor` is NOT excluded. (2) Clusters with mean firing rate <= 0.5 Hz are removed.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 0.5  # Hz

label = ''.join([chr(c) for c in qdata]).strip()
if label in EXCLUDED_QUALITIES:
    continue

mean_fr = trialdat.mean(axis=(0, 2))
keep_mask = mean_fr > LOW_FR
```

iii. The AI justified the quality list from `findClusters.m` and the 0.5 Hz threshold from `getDefaultParams.m` (`params.lowFR = 0.5`), noting the paper says 1 Hz for specific analyses.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For each spike, the go cue time of its trial is subtracted from the spike's `trialtm` to produce aligned spike times. These are then histogrammed into time bins centered on the go cue.

ii.
```python
aligned_times[t_idx] = trialtm[t_idx] - goCue[tr]
counts, _ = np.histogram(spk_t, bins=time_edges)
```

iii. Matches the reference `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 10ms (DT = 1/100), producing 500 bins over the -2.5 to 2.5s window. No rebinning is applied.

ii.
```python
DT = 1.0 / 100  # 10ms bins
time_edges = np.arange(TMIN, TMAX + DT, DT)
```

iii. The AI chose 10ms based on `WorkingWithDataObjs.m` which uses `dt = 1/100`, rather than `getDefaultParams.m` which uses `dt = 1/200` (5ms).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This is defined by the time axis of the bins, not derived from any raw data variable. It is the center of each time bin in the -2.5 to 2.5s window.

ii.
```python
time_centers = time_edges[:-1] + DT / 2
inp = time_centers.astype(np.float32).reshape(1, -1)
```

iii. The time axis is constructed from the bin edges.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as the centers of the bin edges. No processing of raw data is needed.

ii.
```python
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_centers = time_edges[:-1] + DT / 2
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time centers are the centers of the same bins used for spike histogramming, so alignment is by construction.

ii.
```python
counts, _ = np.histogram(spk_t, bins=time_edges)
# ...
inp = time_centers.astype(np.float32).reshape(1, -1)
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Four per-trial fields: `bp.hit`, `bp.miss`, `bp.R`, and `bp.L`. The lick direction is inferred from the combination of correct direction (R/L) and outcome (hit/miss).

ii.
```python
hit = f['obj/bp/hit'][:].flatten().astype(bool)
miss = f['obj/bp/miss'][:].flatten().astype(bool)
R = f['obj/bp/R'][:].flatten().astype(bool)
L = f['obj/bp/L'][:].flatten().astype(bool)
```

iii. R/L indicate the correct direction, not the actual lick direction, so actual lick is derived from the combination.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Hit + R = licked right (1), Hit + L = licked left (0), Miss + R = licked left (0), Miss + L = licked right (1), otherwise no lick (2).

ii.
```python
lick_dir = np.full(ntrials, 2, dtype=int)  # default: none
lick_dir[hit & R] = 1   # correct right -> licked right
lick_dir[hit & L] = 0   # correct left -> licked left
lick_dir[miss & R] = 0  # correct right, wrong -> licked left
lick_dir[miss & L] = 1  # correct left, wrong -> licked right
```

iii. Documented in CONVERSION_NOTES.md as encoding actual lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater`. Autowater=true indicates WC (water-cued) context; false indicates DR (delayed-response).

ii.
```python
autowater = f['obj/bp/autowater'][:].flatten().astype(bool)
```

iii. Documented in CONVERSION_NOTES.md.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct mapping: autowater=true -> WC (0), autowater=false -> DR (1).

ii.
```python
context = np.zeros(ntrials, dtype=int)
context[~autowater] = 1  # DR
context[autowater] = 0   # WC
```

iii. Matches the instruction specification.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit` and `bp.miss`. Trials that are neither hit nor miss are classified as ignore.

ii.
```python
hit = f['obj/bp/hit'][:].flatten().astype(bool)
miss = f['obj/bp/miss'][:].flatten().astype(bool)
```

iii. The three outcome classes are mutually exclusive.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct mapping: hit -> correct (1), miss -> incorrect (0), neither -> ignore (2).

ii.
```python
outcome = np.full(ntrials, 2, dtype=int)  # default: ignore
outcome[hit] = 1  # correct
outcome[miss] = 0  # incorrect
```

iii. Matches the instruction specification.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking in `obj.traj`. Only the **side camera** (index 0) tongue feature is used. The bottom camera tongue (`top_tongue`) is NOT used. Frame times from `traj.frameTimes` and video offset from `sglx.bitcode` are also used.

ii.
```python
tongue_idx = None
for j, name in enumerate(side_feats):
    if name == 'tongue':
        tongue_idx = j
        break
# Only side camera tongue is extracted
tongue_x = ts[tongue_idx, 0, :]
tongue_y = ts[tongue_idx, 1, :]
tongue_conf = ts[tongue_idx, 2, :]
```

iii. The AI only uses the side camera for tongue tracking. No mention of combining two camera views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Five steps: (1) Extract x,y coordinates for the tongue from the side camera. (2) Compute velocity as `sqrt(dx^2 + dy^2)` using simple `np.diff` divided by the median frame interval. No Gaussian smoothing of x,y positions is applied. (3) Set velocity to NaN where confidence < 0.9. (4) Interpolate valid velocity values to the time centers using `np.interp`. (5) Discretize at session 50th percentile.

ii.
```python
def compute_velocity_from_xy(x, y, frame_times, dt_frames):
    dx = np.diff(x) / dt_frames
    dy = np.diff(y) / dt_frames
    vel = np.sqrt(dx**2 + dy**2)
    vel = np.concatenate([vel, [vel[-1] if len(vel) > 0 else 0]])
    return vel

vel = compute_velocity_from_xy(tongue_x, tongue_y, aligned_ft, dt_frames)
vel[tongue_conf < 0.9] = np.nan
tongue_vel[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], vel[valid], ...)
```

iii. The AI does not smooth x,y before differentiation, uses constant dt_frames (median) rather than actual frame intervals, and interpolates rather than bin-averaging.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Discretized at the session 50th percentile (median) of non-NaN values from valid trials. Below threshold = 0, at or above = 1, NaN = 2 (not visible).

ii.
```python
def discretize_velocity(vel_all, valid_trial_indices, not_visible_val=2):
    valid_data = vel_all[:, valid_trial_indices]
    flat_valid = valid_data[~np.isnan(valid_data)]
    threshold = np.median(flat_valid)
    disc[valid_mask & (trial_vel < threshold)] = 0
    disc[valid_mask & (trial_vel >= threshold)] = 1
```

iii. Matches instruction specification.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by the video offset and go cue time, then the velocity is linearly interpolated onto the time centers (same grid as neural data). The video offset is computed as `nanmedian(sglx.bitcode.bitstart/fs) - nanmedian(bp.ev.bitStart)`.

ii.
```python
aligned_ft = frame_times - vidshift - goCue[tr_idx]
tongue_vel[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], vel[valid],
                                   left=np.nan, right=np.nan)
```

iii. The video offset computation uses `nanmedian` rather than `mode`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DLC tracking from the **bottom camera** (index 1). All features with 'paw' in the name (case-insensitive) are used, which includes both `top_paw` and `bottom_paw`.

ii.
```python
paw_indices = [j for j, name in enumerate(bottom_feats) if 'paw' in name.lower()]
```

iii. The AI uses all paw features rather than selecting one specific paw.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same velocity computation as tongue: simple diff/median-dt, confidence threshold at 0.9. Velocities from all paw features are averaged with `np.nanmean`. The result is interpolated to time centers and discretized at session median.

ii.
```python
for pidx in paw_indices:
    vel = compute_velocity_from_xy(px, py, aligned_ft, dt_frames)
    vel[pc < 0.9] = np.nan
    paw_vels.append(vel)
avg_vel = np.nanmean(paw_vels, axis=0)
paw_vel[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], avg_vel[valid], ...)
```

iii. Averaging multiple paw features is a different choice from the reference, which uses only `top_paw`.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: discretized at session 50th percentile. Below = 0, above = 1, NaN = 2.

ii.
```python
paw_disc = discretize_velocity(paw_vel_all, valid_trial_indices, not_visible_val=2)
```

iii. Matches instruction specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times corrected by video offset and go cue, then linearly interpolated to time centers.

ii.
```python
aligned_ft = frame_times - vidshift - goCue[tr_idx]
paw_vel[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], avg_vel[valid], ...)
```

iii. Same approach as tongue alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Separate `motionEnergy_<anm>_<date>.mat` files. The AI handles two formats: struct with `data` field, and direct array.

ii.
```python
me_data = sio.loadmat(me_fpath, squeeze_me=False)
me_raw = me_data['me']
if hasattr(me_raw, 'dtype') and me_raw.dtype.names and 'data' in me_raw.dtype.names:
    me_trials = me_raw['data'][0, 0]
elif me_raw.dtype == object:
    me_trials = me_raw
```

iii. Documented in CONVERSION_NOTES.md.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The per-trial motion energy trace (one value per camera frame) is aligned using the video offset and go cue, then **linearly interpolated** to time centers. NaN values are then **filled with nearest-neighbor interpolation**. Finally discretized at session median.

ii.
```python
me_all[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], me_trial[valid])

# Fill NaNs with nearest
for tr_idx in range(ntrials):
    col = me_all[:, tr_idx]
    nans = np.isnan(col)
    if nans.any() and not nans.all():
        col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), col[~nans])
```

iii. The NaN-filling is a choice that differs from the reference, which leaves NaN bins as "no video" class.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as tongue/paw: discretized at session 50th percentile. Below = 0, above = 1, NaN = 2 (no video).

ii.
```python
me_disc = discretize_velocity(me_all, valid_trial_indices, not_visible_val=2)
```

iii. However, because NaN values were filled by nearest-neighbor interpolation before discretization, fewer bins will actually get the "no video" class compared to the reference.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from the side camera are corrected by video offset and go cue, then motion energy values are linearly interpolated to time centers.

ii.
```python
aligned_ft = frame_times - vidshift - goCue[tr_idx]
me_all[:, tr_idx] = np.interp(time_centers, aligned_ft[valid], me_trial[valid])
```

iii. Uses interpolation rather than bin-averaging.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches: (1) If a session has too few valid trials (<2) or too few neurons after filtering (<10), it is skipped entirely. (2) For DLC data, if loading fails, the entire trial's velocity is left as NaN (becomes "not visible"). (3) For motion energy, NaN values are filled with nearest-neighbor interpolation. (4) Spikes outside the time window or with invalid trial numbers get NaN and are excluded. (5) Broad try/except blocks silently skip individual trials or features on error.

ii.
```python
if len(valid_trial_indices) < 2:
    return None
if keep_mask.sum() < 10:
    return None

# Fill NaNs with nearest (motion energy)
col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), col[~nans])
```

iii. The AI's use of broad try/except blocks may silently mask data issues. The NaN-filling for motion energy fabricates values where data is genuinely missing.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the .mat files dominates runtime. The spike binning loop (per neuron, per trial) is also slow because it iterates over every neuron and every trial in nested Python loops rather than vectorizing.

ii.
```python
for i in range(n_neurons_raw):
    for tr_idx in range(ntrials):
        mask = spk_trials == trial_num
        counts, _ = np.histogram(spk_t, bins=time_edges)
        trialdat[:, i, tr_idx] = smooth_with_bc(fr, kernel, BOUNDARY_CONDITION)
```

iii. Full conversion runs in ~132s per CONVERSION_NOTES.md.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning has a nested loop over neurons and trials that could be vectorized using `np.histogram2d` (as the reference does). The smoothing loop over columns in `smooth_with_bc` could use scipy's `gaussian_filter1d`. The velocity computation loop per trial could potentially be partially vectorized.

ii.
```python
# Nested loop: per neuron, per trial
for i in range(n_neurons_raw):
    for tr_idx in range(ntrials):
        mask = spk_trials == trial_num
        counts, _ = np.histogram(spk_t, bins=time_edges)
```

iii. The reference vectorizes spike binning with a single `np.histogram2d` call per cluster.

## 11-c. What processing does the code repeat multiple times?

i. The video offset is computed separately in the DLC loading function and again in the motion energy loading function. The behavioral field loading (hit, miss, R, L, etc.) is done within each format-specific function rather than being shared.

ii.
```python
# Video offset computed in load_dlc_velocities_h5:
vidshift = sglx_bitstart / sglx_fs - bitStart

# Video offset computed again in load_motion_energy:
vidshift = sglx_bitstart / sglx_fs - bitStart
```

iii. The code has separate h5 and v5 processing paths that duplicate much logic.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `bp.no` (no-response indicator) but never uses it - outcome is derived from hit/miss only. The code loads `bp.L` but it's redundant since `L = ~R` for the relevant trials. The causal Gaussian kernel computation with reflect boundary handling is more complex than needed.

ii.
```python
no_resp = f['obj/bp/no'][:].flatten().astype(bool)  # loaded but never used
L = f['obj/bp/L'][:].flatten().astype(bool)  # redundant with R
```

iii. Minor unnecessary work; does not affect correctness.
