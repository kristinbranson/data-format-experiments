# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers session files by globbing `data_structure_*.mat` in both `Ephys_Behavior/` and `RandomizedDelay_Ephys_Behavior/` directories, then filters against a hard-coded `PROBE_MAP` dictionary that lists the 44 sessions and their probe assignments. Two wrapper classes (`H5Session` and `V5Session`) handle the two MATLAB file formats. Motion energy files are located by constructing the filename from the session key.

ii.
```python
def get_session_list():
    data_dirs = [
        '/app/data/Ephys_Behavior/',
        '/app/data/RandomizedDelay_Ephys_Behavior/',
    ]
    for dpath in data_dirs:
        files = sorted(glob.glob(os.path.join(dpath, 'data_structure_*.mat')))
        for fp in files:
            fn = os.path.basename(fp)
            key = fn.replace('data_structure_', '').replace('.mat', '')
            if key not in PROBE_MAP:
                continue
            ...
```

```python
def open_session(filepath):
    if is_h5_format(filepath):
        return H5Session(filepath)
    else:
        return V5Session(filepath)
```

iii. The AI documented in CONVERSION_NOTES.md that session names and probe assignments were taken from the authors' loading scripts (`load<ANM>_ALMVideo.m`). Sessions not in `PROBE_MAP` are excluded, matching the authors' analysis.

## 1-b. How are the data split into subjects?

i. The subject is extracted from the session key by splitting on underscore and taking the first part (e.g., `JEB19` from `JEB19_2023-04-19`). Subjects are collected into a list in order of first appearance.

ii.
```python
anm = session_key.split('_')[0]
...
anm = result['animal']
if anm not in all_subjects:
    all_subjects.append(anm)
subject_idx.append(all_subjects.index(anm))
```

iii. The AI noted that the animal ID is always encoded in the filename, matching the authors' convention.

## 1-c. How are the data split into sessions?

i. One session corresponds to one `.mat` file, keyed by `<animal>_<date>`. Both `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior` folders are searched. Each session becomes one element of `neural`, `input`, and `output`. 44 sessions total: 25 fixed-delay and 19 randomized-delay.

ii.
```python
PROBE_MAP = {
    'EKH1_2021-08-07': [2], 'EKH3_2021-08-11': [2],
    ...
    'JEB24_2023-11-03': [1],
}
```

iii. The AI documented that these are the sessions listed in the authors' loading scripts.

## 1-d. How are the data split into trials?

i. The number of trials is read from `obj.bp.Ntrials`. The AI iterates over trial numbers 1 to N, matching spikes to trials via the `trial` field in the cluster data. Each trial has one go cue time from `bp.ev.goCue`.

ii.
```python
n_trials = sess.get_ntrials()
...
for j in range(n_trials):
    trial_num = j + 1  # 1-indexed trial number
    spk_mask = trial == trial_num
```

iii. The Bpod table defines trials directly with one go cue per trial.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) photostimulation trials (`stim.enable != 0`) are excluded, (2) early-lick trials (`early != 0`) are excluded, (3) trials with no neural data (recording ended before behavioral session) are excluded.

ii.
```python
valid_mask = (stim_enable == 0) & (early == 0)
valid_trials = np.where(valid_mask)[0]
...
trial_has_spikes = np.any(trialdat > 0, axis=(0, 1))
valid_trials = np.array([t for t in valid_trials if trial_has_spikes[t]])
```

iii. The AI documented that stim trials are excluded because photoinactivation disrupts neural activity, and early-lick trials are excluded per the paper's conventions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu{probe}` spike-sorted clusters, specifically the `trialtm` (spike times relative to trial start), `trial` (trial number for each spike), and `quality` (curation label) fields. The go cue times `bp.ev.goCue` are used for alignment.

ii.
```python
trialtm = unit['trialtm']
trial = unit['trial']
...
spk_times = trialtm[spk_mask] - goCue[j]
counts, _ = np.histogram(spk_times, bins=EDGES)
```

iii. The AI noted these are the standard spike-sorted cluster fields used by the reference code.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue by subtracting `goCue[trial]` from `trialtm`. Aligned spikes are binned into 10ms bins spanning -2.5 to +2.5s (500 bins), converted to firing rate by dividing by dt, then smoothed with a **causal** Gaussian kernel (window=15 bins, first half zeroed, reflect boundary condition). The causal kernel is constructed to match `mySmooth.m` from the reference MATLAB code.

ii.
```python
DT = 1.0/100  # 10 ms bins
EDGES = np.arange(TMIN, TMAX + DT, DT)
...
def make_causal_gaussian_kernel(N):
    kern = gaussian(N, std=(N-1)/(2*2.5))
    kern[:N//2] = 0  # causal
    kern = kern / kern.sum()
    return kern
...
counts, _ = np.histogram(spk_times, bins=EDGES)
rate = counts.astype(np.float32) / DT
trialdat[:, i, j] = smooth_signal(rate)
```

iii. The AI documented that this matches the reference code's `getSeq.m` (binning), `mySmooth.m` (causal Gaussian), and `WorkingWithDataObjs.m` (`dt=1/100`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) clusters with quality labels in `{'garbage', 'gabrga', 'noisy', 'real?'}` are excluded (case-insensitive), (2) units with mean firing rate <= 1 Hz are excluded. Additionally, sessions with fewer than 10 units after filtering are skipped entirely.

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0
MIN_UNITS = 10
...
quality = str(unit['quality'][0]).strip().lower()
if quality in EXCLUDE_QUALITIES:
    continue
...
mean_fr = np.mean(np.mean(trialdat[:, :, valid_trials], axis=2), axis=0)
keep_units = mean_fr > LOW_FR
...
if n_kept < MIN_UNITS:
    print(f"  SKIP: Only {n_kept} units after FR filter")
    return None
```

iii. The AI documented that the quality labels match `findClusters.m` and the FR threshold matches `removeLowFRClusters.m`. The MIN_UNITS=10 filter was noted as matching the paper's session inclusion criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting the go cue time from the spike time within each trial: `trialtm - goCue[trial]`. This matches `alignSpikes.m`.

ii.
```python
spk_times = trialtm[spk_mask] - goCue[j]
counts, _ = np.histogram(spk_times, bins=EDGES)
```

iii. The AI documented this matches the reference's `alignSpikes.m`: `trialtm_aligned = trialtm - event`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 10ms (DT = 1/100), giving 500 time bins over the [-2.5, 2.5]s window. No rebinning is applied - spikes are counted directly into these bins.

ii.
```python
DT = 1.0/100  # 10 ms bins
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME = EDGES[:-1] + DT / 2
N_TIMEBINS = len(TIME)
```

iii. The AI documented this matches `params.dt = 1/100` from `WorkingWithDataObjs.m` in the reference code.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself - the centers of the time bins, computed from the bin edges. It is not derived from any raw data variable but is defined by the binning parameters.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT, DT)
TIME = EDGES[:-1] + DT / 2
...
input_trials.append(TIME.reshape(1, -1).astype(np.float32))
```

iii. The AI noted this matches the reference code's time axis construction.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing - the time axis is simply the bin centers computed from the parameters.

ii.
```python
TIME = EDGES[:-1] + DT / 2
```

iii. N/A - straightforward construction.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input IS the neural time axis - the same bin centers used for spike counting. Each trial gets the same time array since all trials share the same time window.

ii.
```python
input_trials.append(TIME.reshape(1, -1).astype(np.float32))
```

iii. By construction, the input time axis matches the neural binning grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Derived from `bp.R` (right trial indicator), `bp.L` (left trial indicator), `bp.hit` (correct response), `bp.miss` (incorrect response), and `bp.no` (no response/ignore).

ii.
```python
hit = sess.get_bp_field('hit')
miss = sess.get_bp_field('miss')
no = sess.get_bp_field('no')
R = sess.get_bp_field('R')
L = sess.get_bp_field('L')
```

iii. The AI noted that R/L indicate trial type (stimulus side), while hit/miss indicate whether the response was correct, so the actual lick direction must be inferred.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI infers the actual lick direction from the trial type and outcome: right trial + hit = right lick (1), right trial + miss = left lick (0), left trial + hit = left lick (0), left trial + miss = right lick (1), no response = no lick (2).

ii.
```python
lick_dir = np.full(len(valid_trials), 2, dtype=int)
for vi, ti in enumerate(valid_trials):
    if no[ti] == 1:
        lick_dir[vi] = 2
    elif R[ti] == 1 and hit[ti] == 1:
        lick_dir[vi] = 1  # right lick
    elif R[ti] == 1 and miss[ti] == 1:
        lick_dir[vi] = 0  # left lick
    elif L[ti] == 1 and hit[ti] == 1:
        lick_dir[vi] = 0  # left lick
    elif L[ti] == 1 and miss[ti] == 1:
        lick_dir[vi] = 1  # right lick
```

iii. The AI documented the logic linking trial type and outcome to actual lick direction, and also handles WC trials where R/L still indicates which port had water.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`. Autowater=1 indicates the water-cued (WC) context; autowater=0 indicates the delayed-response (DR) context.

ii.
```python
autowater = sess.get_bp_field('autowater')
...
context = np.array([1 if autowater[ti] == 0 else 0 for ti in valid_trials], dtype=int)
```

iii. The AI documented that autowater marks trials where water was delivered without a cue (WC context).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabeling: autowater=1 → WC (0), autowater=0 → DR (1).

ii.
```python
context = np.array([1 if autowater[ti] == 0 else 0 for ti in valid_trials], dtype=int)
```

iii. Simple binary mapping.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit`, `bp.miss`, and `bp.no`.

ii.
```python
hit = sess.get_bp_field('hit')
miss = sess.get_bp_field('miss')
no = sess.get_bp_field('no')
```

iii. Three mutually exclusive outcome fields directly from the Bpod table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. hit → correct (1), miss → incorrect (0), no → ignore (2).

ii.
```python
outcome = np.full(len(valid_trials), 2, dtype=int)
for vi, ti in enumerate(valid_trials):
    if hit[ti] == 1:
        outcome[vi] = 1
    elif miss[ti] == 1:
        outcome[vi] = 0
    elif no[ti] == 1:
        outcome[vi] = 2
```

iii. Direct mapping matching the instruction specification.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from `obj.traj[0]` (side camera) tracking data: the `ts` array contains x, y, and likelihood for each tracked feature including `tongue`, and `frameTimes` provides the time of each frame. The video offset from `sglx.bitcode.bitstart` and `bp.ev.bitStart` is used for alignment.

ii.
```python
side_feats, _, _, _ = sess.get_traj_data(0, 0)
tongue_idx_side = None
for fi, fn in enumerate(side_feats):
    if fn == 'tongue':
        tongue_idx_side = fi
        break
...
_, ts, frame_times, is_valid = sess.get_traj_data(0, ti)
tx = ts[:, 0, tongue_idx_side].astype(float)
ty = ts[:, 1, tongue_idx_side].astype(float)
```

iii. The AI uses only the side camera's tongue feature, not the bottom camera's `top_tongue`.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Five steps: (1) Identify frames where tongue is visible (x and y not NaN). (2) Fill invisible positions with the mean of all visible positions as baseline. (3) Interpolate x, y to the neural time axis using `np.interp`. (4) Compute velocity as the Euclidean speed of the gradient of x and y. (5) Set velocity to 0 where tongue was not visible.

ii.
```python
vis_raw = ~(np.isnan(tx) | np.isnan(ty))
aligned_times = frame_times - vidshift - goCue[ti]
vis_interp = np.interp(taxis, aligned_times, vis_raw.astype(float)) > 0.5
...
baseline_x = np.nanmean(tx[vis_raw])
baseline_y = np.nanmean(ty[vis_raw])
tx_filled[~vis_raw] = baseline_x
ty_filled[~vis_raw] = baseline_y
tx_i = np.interp(taxis, aligned_times, tx_filled)
ty_i = np.interp(taxis, aligned_times, ty_filled)
vx = np.gradient(tx_i)
vy = np.gradient(ty_i)
speed = np.sqrt(vx**2 + vy**2)
speed[~vis_interp] = 0
```

iii. The AI noted this matches `findVelocity.m` where tongue velocity is set to 0 when not visible, and `setTongueBaselinePosition` for filling NaN positions.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile threshold computed over visible tongue velocity values only. Below threshold → 0, at or above threshold → 1, not visible → 2.

ii.
```python
visible_tongue_vals = tongue_vel[tongue_visible & ~np.isnan(tongue_vel)]
thresh_tongue = np.percentile(visible_tongue_vals, 50)
tongue_vel_disc[valid, vi] = (tongue_vel[valid, vi] >= thresh_tongue).astype(int)
tongue_vel_disc[~vis, vi] = 2  # not visible
```

iii. The threshold is computed per-session on visible values only, matching the instruction's 50th percentile specification.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by the video offset and go cue time, then the tongue positions are interpolated from frame times to the neural time axis using `np.interp`. Velocity is then computed on the interpolated positions.

ii.
```python
aligned_times = frame_times - vidshift - goCue[ti]
tx_i = np.interp(taxis, aligned_times, tx_filled)
ty_i = np.interp(taxis, aligned_times, ty_filled)
```

iii. The AI documented that the video offset is computed matching `findVideoOffset.m` and that positions are interpolated to the neural time axis matching `findPosition.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from the bottom camera (`obj.traj[1]`) tracking data, using all features with 'paw' in the name (both `top_paw` and `bottom_paw`).

ii.
```python
paw_indices_bottom = [fi for fi, fn in enumerate(bottom_feats) if 'paw' in fn]
...
for pidx in paw_indices_bottom:
    px = ts[:, 0, pidx].astype(float)
    py = ts[:, 1, pidx].astype(float)
```

iii. The AI uses all paw-named features from the bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature: (1) Fill NaN positions with nearest valid value via interpolation. (2) Interpolate to neural time axis. (3) Compute velocity as gradient of interpolated positions. (4) Subtract baseline velocity (median of gradient). (5) Compute Euclidean speed. Then average speeds across all paw features.

ii.
```python
valid_idx = np.where(vis_raw)[0]
px_filled = np.interp(np.arange(len(px)), valid_idx, px[valid_idx])
py_filled = np.interp(np.arange(len(py)), valid_idx, py[valid_idx])
px_i = np.interp(taxis, aligned_times, px_filled)
py_i = np.interp(taxis, aligned_times, py_filled)
vx = np.gradient(px_i)
vy = np.gradient(py_i)
base_vx = np.median(np.diff(px_i))
base_vy = np.median(np.diff(py_i))
vx = vx - base_vx
vy = vy - base_vy
speed = np.sqrt(vx**2 + vy**2)
...
paw_vel[:, vi] = np.mean(paw_speeds, axis=0)
```

iii. The AI documented that this matches `findPosition.m` (nearest fill) and `findVelocity.m` (gradient with baseline subtraction).

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Per-session 50th percentile of visible paw values. Below threshold → 0, at or above → 1, not visible → 2.

ii.
```python
visible_paw_vals = paw_vel[paw_visible & ~np.isnan(paw_vel)]
thresh_paw = np.percentile(visible_paw_vals, 50)
paw_vel_disc[valid, vi] = (paw_vel[valid, vi] >= thresh_paw).astype(int)
paw_vel_disc[~vis, vi] = 2
```

iii. Same approach as tongue velocity thresholding.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times corrected by video offset and go cue, positions interpolated to neural time axis.

ii.
```python
aligned_times = frame_times - vidshift - goCue[ti]
px_i = np.interp(taxis, aligned_times, px_filled)
```

iii. Same interpolation approach as all video-based outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Derived from separate `motionEnergy_<anm>_<date>.mat` files. The AI handles three formats: HDF5 v7.3, nested v5 struct, and plain v5 cell array.

ii.
```python
def load_motion_energy(me_filepath):
    if is_h5_format(me_filepath):
        f = h5py.File(me_filepath, 'r')
        me = f['me']
        data_ds = me['data']
        ...
    else:
        d = sio.loadmat(me_filepath, squeeze_me=False)
        me_raw = d['me']
        if me_raw.dtype.names and 'data' in me_raw.dtype.names:
            inner = me_raw[0, 0]['data']
            if hasattr(inner, 'dtype') and inner.dtype.names and 'data' in inner.dtype.names:
                data_arr = inner[0, 0]['data']
                ...
```

iii. The AI documented that three different file formats exist and handled each case, matching `loadMotionEnergy.m`'s guard for nested structs.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values are interpolated from frame times to the neural time axis. NaN values are filled with nearest interpolation.

ii.
```python
me_interp = np.interp(taxis, aligned_times[:n_me], me_trial[:n_me])
nan_mask = np.isnan(me_interp)
if np.any(~nan_mask) and np.any(nan_mask):
    valid_idx = np.where(~nan_mask)[0]
    me_interp = np.interp(np.arange(len(me_interp)), valid_idx, me_interp[valid_idx])
```

iii. The AI documented this matches `loadMotionEnergy.m` which interpolates ME to the neural time axis.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile of valid (non-NaN) values. Below threshold → 0, at or above → 1, no video → 2.

ii.
```python
me_vals = me_data_aligned[valid_me]
thresh_me = np.percentile(me_vals, 50)
me_disc[valid_t, vi] = (me_data_aligned[valid_t, vi] >= thresh_me).astype(int)
me_disc[~valid_t, vi] = 2
```

iii. Same approach as other velocity thresholding.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from the side camera are corrected by video offset and go cue time, then ME values are interpolated from frame times to the neural time axis.

ii.
```python
_, _, frame_times, is_valid = sess.get_traj_data(0, ti)
aligned_times = frame_times - vidshift - goCue[ti]
me_interp = np.interp(taxis, aligned_times[:n_me], me_trial[:n_me])
```

iii. Same interpolation approach as other video-based outputs, using side camera frame times.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple approaches: (1) Trials with no neural data (recording ended early) are excluded. (2) Trajectory data loading failures are caught with broad `except Exception: pass` blocks, leaving default values (NaN for velocity, 0 for visibility). (3) Motion energy loading failures return None. (4) Frame time validity is checked via NdroppedFrames field. (5) Tongue positions with NaN are filled with mean visible position as baseline.

ii.
```python
try:
    _, ts, frame_times, is_valid = sess.get_traj_data(0, ti)
    if is_valid and len(frame_times) > 1 and not np.all(np.isnan(frame_times)):
        ...
except Exception:
    pass
```

iii. The AI documented handling of all-zero trials, ME loading failures, and trajectory data issues in CONVERSION_NOTES.md Step 10.

## 11-a. What are the most time-consuming steps of the code?

i. The spike binning is extremely time-consuming because it uses a nested loop over all units and all trials. For each unit-trial pair, a separate histogram is computed. File loading also contributes significantly.

ii.
```python
for i, unit in enumerate(all_units):
    trialtm = unit['trialtm']
    trial = unit['trial']
    for j in range(n_trials):
        trial_num = j + 1
        spk_mask = trial == trial_num
        ...
        counts, _ = np.histogram(spk_times, bins=EDGES)
        rate = counts.astype(np.float32) / DT
        trialdat[:, i, j] = smooth_signal(rate)
```

iii. The AI did not explicitly document the spike binning as a bottleneck, but it is clearly the most compute-intensive part due to the O(n_units * n_trials) loop.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning nested loop (unit x trial) is the most obvious candidate. The reference solution uses `np.histogram2d` to bin all trials for a given unit in a single call. The AI's approach processes each trial individually with `np.histogram`. Additionally, the smoothing is applied per-unit-per-trial rather than vectorized.

ii.
```python
# AI: per-unit, per-trial loop
for i, unit in enumerate(all_units):
    for j in range(n_trials):
        ...
        counts, _ = np.histogram(spk_times, bins=EDGES)
        trialdat[:, i, j] = smooth_signal(rate)

# Reference (for comparison): vectorized per-unit
counts, _, _ = np.histogram2d(spike_trial, spike_time, bins=[trial_edges, BIN_EDGES])
```

iii. The AI did not document this inefficiency. The per-trial video processing loops are less amenable to vectorization due to variable frame counts.

## 11-c. What processing does the code repeat multiple times?

i. The trajectory data is loaded multiple times per trial: once for tongue processing, once for paw processing, and once for motion energy alignment (to get frame times). The `get_traj_data` call does extensive work parsing the MATLAB structure each time.

ii.
```python
# Called for tongue:
_, ts, frame_times, is_valid = sess.get_traj_data(0, ti)
...
# Called again for motion energy:
_, _, frame_times, is_valid = sess.get_traj_data(0, ti)
```

iii. The AI did not document this repeated loading.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several forms of unnecessary processing: (1) The spike binning computes firing rates for ALL trials (including early-lick and stim trials) before subsetting to valid trials. This wastes computation on trials that are discarded. (2) The `no` field is loaded but not strictly needed since ignore is the default. (3) The AI computes the initial lick_dir array, then immediately recomputes it ("Re-do lick direction properly"). (4) The motion energy threshold (`moveThresh`) is loaded but never used.

ii.
```python
# Computes for ALL trials, then subsets
trialdat = np.zeros((N_TIMEBINS, n_units, n_trials), dtype=np.float32)
for i, unit in enumerate(all_units):
    for j in range(n_trials):  # all n_trials, not just valid
        ...
# Later:
trialdat_valid = trialdat[:, :, valid_trials]
```

```python
# Lick direction computed twice
lick_dir = np.full(len(valid_trials), 2, dtype=int)
for vi, ti in enumerate(valid_trials):
    ...  # first attempt
# Re-do lick direction properly:
lick_dir = np.full(len(valid_trials), 2, dtype=int)
for vi, ti in enumerate(valid_trials):
    ...  # second attempt
```

iii. The AI did not document these inefficiencies in CONVERSION_NOTES.md.
