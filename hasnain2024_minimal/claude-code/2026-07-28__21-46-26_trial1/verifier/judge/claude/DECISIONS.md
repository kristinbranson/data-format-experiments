# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by globbing for `data_structure_*.mat` files in the two data directories (`Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`). Each file is loaded using h5py (HDF5/v7.3) with a fallback to scipy.io (v5/v7). Motion energy is loaded from separate `motionEnergy_*.mat` files matched by name. Sessions without neural data (no `clu` field) are skipped.

ii.
```python
def find_session_files(data_dirs):
    for data_dir in data_dirs:
        data_files = sorted(glob.glob(os.path.join(data_dir, 'data_structure_*.mat')))
        for df in data_files:
            parts = basename.replace('data_structure_', '').replace('.mat', '')
            me_file = os.path.join(data_dir, f'motionEnergy_{parts}.mat')
            sessions.append({'data_file': df, 'me_file': me_file, ...})

def load_session_data(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            ...
        return _load_session_data_h5(filepath)
    except (OSError, ValueError):
        pass
    return load_session_data_v5(filepath)
```

iii. The AI uses glob-based discovery rather than a hardcoded session list. The CONVERSION_NOTES.md notes that 2 sessions were skipped (behavior-only, no neural data). The AI found 47 files total (25 + 22), processing 45 with neural data.

## 1-b. How are the data split into subjects?

i. The subject (animal) is extracted from the filename by splitting `data_structure_ANIMAL_DATE.mat` on underscores, taking the first part as the animal ID.

ii.
```python
parts = basename.replace('data_structure_', '').replace('.mat', '').split('_')
animal = parts[0]
```

iii. The animal ID is taken from the filename, which is consistent with how the reference paper's loading scripts identify animals.

## 1-c. How are the data split into sessions?

i. Each `data_structure_*.mat` file is one session. Sessions from both `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior` directories are included. The AI found 47 files total, processed 45 (2 were behavior-only).

ii.
```python
def find_session_files(data_dirs):
    for data_dir in data_dirs:
        data_files = sorted(glob.glob(os.path.join(data_dir, 'data_structure_*.mat')))
```

iii. The AI discovers sessions by globbing rather than using a hardcoded session list from the authors' loading scripts. This may include sessions the authors excluded.

## 1-d. How are the data split into trials?

i. Trials are defined by the behavioral data structure, with `Ntrials` giving the total count. Each trial has corresponding entries in the behavioral fields (hit, miss, R, L, etc.), spike data (via trial assignments), and trajectory data.

ii.
```python
Ntrials = int(h5_read_scalar(bp['Ntrials']))
session['hit'] = bp['hit'][()].flatten().astype(float)
# ... per-trial fields loaded similarly
```

iii. The trial structure is directly provided by the Bpod behavioral data.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) photostimulation trials excluded (`stim_enable == 0`), (2) early-lick trials excluded (`early == 0`), (3) **only responding trials kept** (`hit | miss`), which excludes ignore/no-response trials. This is more restrictive than the reference, which keeps ignore trials.

ii.
```python
trial_mask = (
    (session['stim_enable'] == 0) &
    (session['early'] == 0) &
    ((session['hit'] == 1) | (session['miss'] == 1))
)
```

iii. The CONVERSION_NOTES state: "Included only responding trials (hit | miss), excluding ignore/no-response trials. This includes both correct (hit) and incorrect (miss) trials for outcome decoding."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu` cluster data: `trialtm` (spike times relative to trial start), `trial` (trial assignment for each spike), and `quality` (curation label). Go cue times from `bp.ev.goCue` are used for alignment.

ii.
```python
unit['trialtm'] = f[tm_ref][()].flatten()
unit['trial'] = f[trial_ref][()].flatten().astype(int)
unit['quality'] = h5_read_string(f, f[q_ref])
```

iii. These are the standard spike-sorted cluster fields from the data structure.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue, binned into 5ms bins from -2.5 to +2.5s, converted to firing rates (Hz), then smoothed with a **causal** Gaussian kernel of width 15 samples. The kernel is created as a full Gaussian window, then the left half is zeroed to make it causal, matching the AI's interpretation of `mySmooth.m`.

ii.
```python
def causal_gaussian_smooth(x, kernel_width_samples):
    N = kernel_width_samples
    n = np.arange(N)
    center = (N - 1) / 2
    sigma = N / 6  # MATLAB gausswin default
    kern = np.exp(-0.5 * ((n - center) / sigma) ** 2)
    kern[:N // 2] = 0  # Make causal
    kern = kern / kern.sum()
    return np.convolve(x, kern, mode='same')
```

iii. The CONVERSION_NOTES state: "Causal Gaussian kernel with width N=15 samples (matching mySmooth.m). Half of kernel zeroed for causality."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Quality filter excluding 'garbage', 'gabrga', 'noisy', 'real?' labels (but NOT 'poor'). Empty/unlabeled units are included. (2) Firing rate filter removing units with mean FR < 0.5 Hz (rather than the reference's 1.0 Hz). Additionally, probes are filtered by location, keeping only ALM probes.

ii.
```python
excluded = {'garbage', 'gabrga', 'noisy', 'real?'}
# ...
fr_mask = mean_frs > LOW_FR  # LOW_FR = 0.5
# ...
if 'ALM' not in loc_upper:
    print(f"  Probe {probe_idx} ({loc}): skipping non-ALM probe")
    continue
```

iii. The CONVERSION_NOTES state: "Quality filter: excluded 'garbage', 'noisy' units (matching findClusters.m with quality='all'). Firing rate filter: removed units with mean FR < 0.5 Hz (matching removeLowFRClusters.m, default lowFR=0.5)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting the go cue time from each spike's `trialtm`. The aligned times are then binned into histogram bins from -2.5 to +2.5s.

ii.
```python
aligned_times = spike_times[spike_mask] - go_time
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. This matches the standard alignment approach described in the reference code's `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms bins (dt = 1/200 s), giving 1000 time bins per trial from -2.5 to +2.5 s around the go cue. No rebinning is applied.

ii.
```python
DT = 1 / 200  # 5 ms bins
TMIN = -2.5
TMAX = 2.5
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. Matches `getDefaultParams.m` with `params.dt = 1/200` and `params.tmin/tmax`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the center of each 5ms time bin, spanning -2.4975 to +2.4975 s. It is defined from the bin grid, not from any raw data variable.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2
input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
```

iii. The time axis is constructed from the bin edges.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing; the time axis is defined directly from the bin grid parameters.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is the same bin grid used for spike binning, so they are inherently aligned.

ii.
```python
input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
```

iii. Same bin edges are used for both.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction directly from `obj.bp.R`, treating it as the actual lick direction rather than the instructed side.

ii.
```python
lick_direction = session['R'][valid_trials].astype(np.int32)  # R=1, L=0
```

iii. The CONVERSION_NOTES state: "Lick direction: R=1 (right), L=0 (left) from obj.bp.R field."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. `bp.R` is used directly as lick direction (R=1 right, L=0 left). Since ignore trials are already filtered out, only hit and miss trials remain. However, `bp.R` is the **instructed** side, not the actual lick direction. On miss trials, the animal licked the **opposite** of the instructed side, so using `bp.R` directly gives the wrong lick direction for miss trials. The AI has only 2 classes (left=0, right=1) since ignore trials are excluded.

ii.
```python
lick_direction = session['R'][valid_trials].astype(np.int32)  # R=1, L=0
```

iii. The AI treats `bp.R` as the actual lick direction, but this is the instructed side. On miss trials, the lick direction should be opposite to the instructed side.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `obj.bp.autowater`. Autowater=1 indicates WC context, otherwise DR context.

ii.
```python
context = (1 - session['autowater'][valid_trials]).astype(np.int32)  # DR=1, WC=0
```

iii. Consistent with the reference approach.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A simple inversion: `1 - autowater` gives DR=1, WC=0, matching the instruction's coding.

ii.
```python
context = (1 - session['autowater'][valid_trials]).astype(np.int32)
```

iii. Matches the required coding: WC=0, DR=1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `obj.bp.hit` directly. Since ignore trials are already filtered out, hit=1 means correct, and the remaining (miss) trials are incorrect=0.

ii.
```python
outcome = session['hit'][valid_trials].astype(np.int32)  # correct=1, incorrect=0
```

iii. Works because ignore trials were already excluded in the trial filter.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct use of the `hit` field: hit=1 maps to correct=1, hit=0 (which equals miss, since ignores were excluded) maps to incorrect=0. This gives only 2 classes vs the reference's 3 classes (which includes ignore=2).

ii.
```python
outcome = session['hit'][valid_trials].astype(np.int32)
```

iii. The CONVERSION_NOTES state: "Outcome: correct=1, incorrect=0 from obj.bp.hit."

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from `obj.traj` trajectory data, specifically the 'tongue' feature from the side camera (cam 0) only. The reference uses both side camera ('tongue') and bottom camera ('top_tongue').

ii.
```python
tongue_vel = extract_kinematic_feature(session, 0, 'tongue', valid_trials, go_cue_times)
```

iii. The CONVERSION_NOTES state: "Tongue velocity: Extracted from side camera (cam 0), 'tongue' feature."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial: (1) Extract x,y positions for the tongue feature. (2) Compute velocity as `sqrt(dx^2 + dy^2)` using `np.gradient` with a fixed dt of 1/400s. (3) Set velocity to 0 where tongue is not visible (NaN positions). (4) Interpolate to the neural time axis using `scipy.interpolate.interp1d`. (5) Fill remaining NaN with nearest valid values, or 0 if all NaN. No Gaussian smoothing of positions, no likelihood-based filtering, no normalization between cameras (only one camera used).

ii.
```python
dx = np.gradient(x_pos, dt_vid)
dy = np.gradient(y_pos, dt_vid)
vel = np.sqrt(dx**2 + dy**2)
if is_tongue:
    nan_mask = np.isnan(x_pos) | np.isnan(y_pos)
    vel[nan_mask] = 0.0
interp_fn = interp1d(aligned_frame_times, vel, kind='linear', bounds_error=False, fill_value=np.nan)
velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. The AI notes this matches `findPosition.m`, but several processing details differ from the reference: no position smoothing, no likelihood threshold filtering, gradient computed with fixed dt rather than actual frame times, and interpolation to bin centers rather than binning by averaging frames within each bin.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Discretized into 2 bins using the session-wide 50th percentile threshold. Values >= threshold get class 1, values < threshold get class 0. No "not visible" third class.

ii.
```python
def discretize_per_session(values, percentile=50):
    threshold = np.percentile(valid, percentile)
    discretized = (values >= threshold).astype(np.int32)
    return discretized
```

iii. The CONVERSION_NOTES note that tongue velocity becomes degenerate: "The 50th percentile threshold is 0, making all values >= 0 classify as 'high' (class 1)" because the tongue is invisible most of the time and invisible frames are set to 0.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by the video offset (`vidshift`) and go cue time, then the velocity is interpolated to the neural time axis using linear interpolation.

ii.
```python
aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
interp_fn = interp1d(aligned_frame_times, vel, kind='linear', bounds_error=False, fill_value=np.nan)
velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. The video offset is computed using median (not mode as in the reference's `findVideoOffset.m`).

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from both `top_paw` and `bottom_paw` features from the bottom camera (cam 1), averaged together when both are available.

ii.
```python
paw_top = extract_kinematic_feature(session, 1, 'top_paw', valid_trials, go_cue_times)
paw_bot = extract_kinematic_feature(session, 1, 'bottom_paw', valid_trials, go_cue_times)
if paw_top is not None and paw_bot is not None:
    paw_vel = (paw_top + paw_bot) / 2.0
```

iii. The reference uses only `top_paw`, reasoning that `bottom_paw` has unreliable tracking during the delay epoch.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same velocity computation as tongue, but for non-tongue features, missing (NaN) positions are filled with nearest valid values before computing velocity. The two paw velocities are averaged. Then discretized into 2 bins at the 50th percentile.

ii.
```python
if not is_tongue:
    for pos in [x_pos, y_pos]:
        mask = np.isnan(pos)
        if mask.any() and not mask.all():
            valid_idx = np.where(~mask)[0]
            pos[mask] = np.interp(np.where(mask)[0], valid_idx, pos[valid_idx])
```

iii. The AI applies nearest-fill for paw positions before velocity computation, treating the paw differently from the tongue.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: discretized into 2 bins at the session 50th percentile. No "not visible" class.

ii.
```python
paw_vel_disc = discretize_per_session(paw_vel)
```

iii. Follows the instruction specification of 2 bins at the 50th percentile.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times corrected by video offset and go cue, then interpolated to neural time axis.

ii.
```python
aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
interp_fn = interp1d(aligned_frame_times, vel, kind='linear', bounds_error=False, fill_value=np.nan)
```

iii. Same alignment approach as all kinematic features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from separate `motionEnergy_*.mat` files. The data is a per-trial array of frame-level motion energy values.

ii.
```python
def load_motion_energy(filepath):
    mat = scipy.io.loadmat(filepath, squeeze_me=False)
    me = mat['me']
    data_field = me['data'][0, 0]
    me_data = []
    for i in range(data_field.shape[0]):
        for j in range(data_field.shape[1]):
            trial_me = data_field[i, j].flatten().astype(float)
            me_data.append(trial_me)
```

iii. The AI loads the motion energy with scipy.io, falling back to h5py. The CONVERSION_NOTES indicate that 8 sessions had unloadable motion energy files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is interpolated from frame times to the neural time axis using linear interpolation, then discretized into 2 bins at the session 50th percentile. Missing values are filled with nearest valid values or 0.

ii.
```python
interp_fn = interp1d(aligned_frame_times, me_trial,
                    kind='linear', bounds_error=False, fill_value=np.nan)
me_interp[:, ti] = interp_fn(TIME_AXIS)
```

iii. The reference bins motion energy by averaging frames within each 5ms bin rather than interpolating.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Discretized into 2 bins at the session 50th percentile. No "not visible" class.

ii.
```python
me_disc = discretize_per_session(me_interp)
```

iii. Follows instruction specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from the side camera (cam 0) trajectory data are used, corrected by video offset and go cue. The motion energy is interpolated to the neural time axis.

ii.
```python
if len(session['traj']) > 0 and trial_idx < len(session['traj'][0]['trials']):
    ft = session['traj'][0]['trials'][trial_idx].get('frameTimes')
aligned_frame_times = frame_times - vidshift - go_time
interp_fn = interp1d(aligned_frame_times, me_trial, ...)
```

iii. Uses the same video offset correction as other camera streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches: (1) Missing frame times: if frameTimes is None or all NaN, synthetic frame times are generated at 400 Hz. (2) Missing tongue positions: velocity set to 0. (3) Missing paw positions: filled with nearest valid value. (4) Missing motion energy files: 8 sessions set to all zeros. (5) NaN values in interpolated outputs: filled with nearest valid value, or 0 if entirely NaN. (6) Behavior-only sessions (no `clu` field): skipped entirely.

ii.
```python
if frame_times is None or np.all(np.isnan(frame_times)):
    frame_times = np.arange(1, n_frames + 1) / 400.0
# ...
for ti in range(n_trials):
    col = velocities[:, ti]
    mask = np.isnan(col)
    if mask.any() and not mask.all():
        valid_idx = np.where(~mask)[0]
        col[mask] = np.interp(np.where(mask)[0], valid_idx, col[valid_idx])
    elif mask.all():
        velocities[:, ti] = 0.0
```

iii. The AI fills in missing data with synthetic or interpolated values rather than preserving it as a "not visible" class, which differs from the reference approach.

## 11-a. What are the most time-consuming steps of the code?

i. Loading each session's MATLAB file is the most time-consuming step, especially the HDF5 format files which require dereferencing many object references. The per-unit, per-trial spike binning loop is also expensive since it processes each unit and trial individually rather than vectorizing.

ii.
```python
for ui, ci in enumerate(cluster_indices):
    for ti, trial_idx in enumerate(valid_trials):
        spike_mask = spike_trials == trial_num
        aligned_times = spike_times[spike_mask] - go_time
        counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The double loop over units and trials for spike binning is particularly costly compared to the reference's vectorized `histogram2d` approach.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning has a double loop over units and trials that could be vectorized using `np.histogram2d` (as the reference does). The kinematic feature extraction also loops per-trial but this is harder to vectorize due to varying frame counts. The causal smoothing loops over columns when the data is 2D.

ii.
```python
for ui, ci in enumerate(cluster_indices):
    for ti, trial_idx in enumerate(valid_trials):
        # per-unit, per-trial histogram
        counts, _ = np.histogram(aligned_times, bins=edges)
        fr_smooth = causal_gaussian_smooth(fr, smooth_samples)
```

iii. The spike binning is the most impactful loop that could be vectorized.

## 11-c. What processing does the code repeat multiple times?

i. The `spike_mask = spike_trials == trial_num` comparison is repeated for every unit-trial combination, scanning the full spike array each time. Frame times are re-read for each kinematic feature extraction rather than being cached. The trajectory data structure is re-accessed for each feature.

ii.
```python
for ti, trial_idx in enumerate(valid_trials):
    spike_mask = spike_trials == trial_num  # repeated for every trial
```

iii. Caching trial-specific spike masks or pre-sorting spikes by trial would avoid repeated scans.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads many fields from the MATLAB files that are never used (e.g., `sample` event times, `delay` event times, `NdroppedFrames`, `no` field). The `bottom_paw` velocity is computed but could be avoided if only `top_paw` were used (as the reference does). The probe location metadata is read and processed but all probes end up being ALM anyway. The AI also reads `L` (left instruction) which is redundant with `R`.

ii.
```python
session['sample'] = ev['sample'][()].flatten() if 'sample' in ev else np.zeros(Ntrials)
session['delay'] = ev['delay'][()].flatten() if 'delay' in ev else np.zeros(Ntrials)
session['no'] = bp['no'][()].flatten().astype(float) if 'no' in bp.dtype.names else np.zeros(Ntrials)
session['L'] = bp['L'][0, 0].flatten().astype(float)
```

iii. These fields are loaded but never used in the final output construction.
