# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by globbing for `data_structure_*.mat` files in both `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior` directories, rather than using a hard-coded session list. Each session file is tried first as HDF5 (v7.3) using `h5py`, falling back to `scipy.io.loadmat` for v5/v7 format. Motion energy is loaded from a corresponding `motionEnergy_*.mat` file. Each file format has a dedicated loader function (`_load_session_data_h5` and `load_session_data_v5`).

ii.
```python
def find_session_files(data_dirs):
    """Find all data_structure and motionEnergy file pairs."""
    sessions = []
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            continue
        data_files = sorted(glob.glob(os.path.join(data_dir, 'data_structure_*.mat')))
        for df in data_files:
            basename = os.path.basename(df)
            parts = basename.replace('data_structure_', '').replace('.mat', '')
            me_file = os.path.join(data_dir, f'motionEnergy_{parts}.mat')
            if not os.path.exists(me_file):
                me_file = None
            sessions.append({
                'data_file': df,
                'me_file': me_file,
                'animal_date': parts,
            })
    return sessions
```

```python
def load_session_data(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            if 'obj' not in f:
                raise ValueError("No obj field")
            obj = f['obj']
            if 'clu' not in obj:
                raise ValueError("No clu field - behavior-only session")
        return _load_session_data_h5(filepath)
    except (OSError, ValueError) as e:
        pass
    return load_session_data_v5(filepath)
```

iii. The AI noted during development: "Let me start by exploring the project structure and understanding the data and code." It discovered session files by globbing rather than consulting the authors' loading scripts. The trajectory shows the AI encountered errors with some sessions (JEB24_2023-10-03, JEB24_2023-10-04 which lack `clu` fields) and handled them with try/except blocks.

## 1-b. How are the data split into subjects?

i. The subject (animal) name is extracted from the filename by splitting `data_structure_ANIMAL_DATE.mat` and taking the first part. The unique sorted set of animals becomes the `subjects` list.

ii.
```python
basename = os.path.basename(session_file)
parts = basename.replace('data_structure_', '').replace('.mat', '').split('_')
animal = parts[0]
```

```python
unique_subjects = sorted(set(all_animals))
subject_idx = np.array([unique_subjects.index(a) for a in all_animals], dtype=np.int64)
```

iii. The AI extracts the animal name from the filename, which is consistent with the data naming convention.

## 1-c. How are the data split into sessions?

i. Each `data_structure_*.mat` file found by globbing constitutes one session. Both `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior` directories are searched. Sessions without a `clu` field (behavior-only) are skipped. The AI reports 45 sessions were processed.

ii.
```python
DATA_DIRS = [
    '/app/data/Ephys_Behavior',
    '/app/data/RandomizedDelay_Ephys_Behavior',
]
```

iii. The AI's trajectory shows: "45 sessions successfully processed. The 2 failed sessions (JEB24_2023-10-03/04) are HDF5 format but lack 'clu' (behavior-only sessions)."

## 1-d. How are the data split into trials?

i. Each trial corresponds to one entry in the behavioral data arrays (hit, miss, R, L, etc.) up to `Ntrials`. Spike data carries trial assignments (`clu.trial`), and per-trial behavioral variables are indexed directly from the session's behavioral arrays.

ii.
```python
Ntrials = int(h5_read_scalar(bp['Ntrials']))
session['Ntrials'] = Ntrials
session['R'] = bp['R'][()].flatten().astype(float)
# ... etc for each behavioral variable
```

iii. The AI reads the trial count from `Ntrials` and uses it to determine array lengths. This matches the standard approach of using the Bpod table's trial structure.

## 1-e. How are trials filtered based on quality controls?

i. Three filters: early-lick trials (`early == 0`), photostimulation trials (`stim_enable == 0`), and only responding trials (`hit == 1 | miss == 1`). The third filter removes ignore trials entirely, which differs from the reference which keeps ignore trials. No filter is applied for trials past the end of the recording.

ii.
```python
trial_mask = (
    (session['stim_enable'] == 0) &
    (session['early'] == 0) &
    ((session['hit'] == 1) | (session['miss'] == 1))
)
```

iii. The AI's trajectory states: "Include: non-stim, non-early, responding (hit or miss) trials." This removes ignore trials from the dataset entirely. The reference keeps ignore trials, assigning them an "ignore" outcome class and a "no lick" direction.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The spike-sorted cluster data: `clu.trialtm` (spike times relative to trial start), `clu.trial` (trial assignment), and `clu.quality` (curation label). Go cue times `bp.ev.goCue` are used for alignment.

ii.
```python
unit['trialtm'] = f[tm_ref][()].flatten()
unit['trial'] = f[trial_ref][()].flatten().astype(int)
# ...
aligned_times = spike_times[spike_mask] - go_time
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The AI correctly identified the spike data fields from exploring the MATLAB file structure.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue, binned into 5 ms bins (-2.5 to +2.5 s), converted to firing rates (counts/dt), then smoothed with a **causal** Gaussian kernel. The causal kernel is constructed as a `gausswin(15)` with the left half zeroed out, matching `mySmooth.m`. This differs from the reference solution which uses a symmetric (non-causal) Gaussian with 14 ms sigma.

ii.
```python
def causal_gaussian_smooth(x, kernel_width_samples):
    N = kernel_width_samples
    n = np.arange(N)
    center = (N - 1) / 2
    sigma = N / 6  # MATLAB gausswin default: alpha=2.5, sigma = (N-1)/(2*alpha)
    kern = np.exp(-0.5 * ((n - center) / sigma) ** 2)
    kern[:N // 2] = 0
    kern = kern / kern.sum()
    # ...
    return np.convolve(x, kern, mode='same')
```

```python
SMOOTH_MS = 15  # causal Gaussian smoothing kernel width in ms
fr = counts / dt
fr_smooth = causal_gaussian_smooth(fr, smooth_samples)
```

iii. The AI noted "causal Gaussian smoothing matching mySmooth.m." The MATLAB code `mySmooth.m` does zero the left half of the kernel. However, the AI passes `SMOOTH_MS = 15` as the kernel width in **samples** (not ms), and the kernel sigma calculation uses `N/6` rather than `(N-1)/(2*2.5)`. Additionally, the smoothing is applied per-trial per-unit in a Python loop rather than vectorized.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters are filtered by quality label, excluding `{'garbage', 'gabrga', 'noisy', 'real?'}` (matching `findClusters.m`). Empty quality strings are included. Then units with mean firing rate <= 0.5 Hz are removed. The reference uses a threshold of 1.0 Hz. Also, the AI does not include `'poor'` in the exclusion set.

ii.
```python
excluded = {'garbage', 'gabrga', 'noisy', 'real?'}
# ...
LOW_FR = 0.5  # minimum firing rate threshold (Hz)
fr_mask = mean_frs > LOW_FR
```

iii. The AI's reasoning: "Filter by quality (matching findClusters.m with quality='all')". The 0.5 Hz threshold is lower than the paper's stated 1 Hz cutoff ("all units with firing rates exceeding 1 Hz were included in all other analyses").

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting the go cue time: `aligned_times = spike_times[spike_mask] - go_time`. This is the same approach as the reference.

ii.
```python
go_time = go_cue_times[trial_idx]
spike_mask = spike_trials == trial_num
aligned_times = spike_times[spike_mask] - go_time
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The AI correctly aligns spikes to the go cue by subtraction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms bins from -2.5 to +2.5 s, matching the reference's `params.dt = 1/200` and `params.tmin`/`params.tmax`. The bin edges are computed as `np.arange(TMIN, TMAX + DT/2, DT)`.

ii.
```python
DT = 1 / 200  # 5 ms bins
TMIN = -2.5
TMAX = 2.5
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
N_TIMEBINS = len(TIME_AXIS)
```

iii. The AI correctly matches the reference's temporal parameters. Note: using `np.arange` with floating-point step can produce slightly different edge counts than the reference's `int(round(...))` approach, but in practice 1000 bins are produced.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself -- the centers of the 5 ms bins spanning -2.5 to +2.5 s from the go cue. It is not derived from raw data variables but defined by the binning parameters.

ii.
```python
TIME_AXIS = EDGES[:-1] + DT / 2
# ...
input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
```

iii. Same approach as the reference.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing -- the time axis is directly constructed from the bin parameters.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis is the same binning grid used for spike counting, so alignment is inherent.

ii.
```python
input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
```

iii. Same approach as the reference.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI uses only `bp.R` (right-instructed side) to determine lick direction, directly treating it as the animal's lick direction. It does not use `bp.hit` and `bp.miss` to infer actual lick direction.

ii.
```python
lick_direction = session['R'][valid_trials].astype(np.int32)  # R=1, L=0
```

iii. The AI treats R=1 as "right lick" and R=0 as "left lick" without considering outcome. Since ignore trials are already filtered out, this is correct for hit trials but incorrect for miss trials (where the animal licked the opposite direction from the instructed side).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A direct cast of the `R` field to integer. No "no lick" class exists because ignore trials are excluded. This produces only two classes (left=0, right=1) instead of the reference's three (left=0, right=1, no lick=2).

ii.
```python
lick_direction = session['R'][valid_trials].astype(np.int32)  # R=1, L=0
```

```python
'output_values': [
    ['left', 'right'],
    ...
]
```

iii. The AI does not derive lick direction from the combination of instructed side and outcome, so miss trials have the wrong lick direction assigned.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The `bp.autowater` field. Autowater trials are WC context (0), non-autowater are DR context (1).

ii.
```python
context = (1 - session['autowater'][valid_trials]).astype(np.int32)  # DR=1, WC=0
```

iii. This matches the reference approach.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A simple inversion: `1 - autowater` maps autowater=1 to WC=0 and autowater=0 to DR=1.

ii.
```python
context = (1 - session['autowater'][valid_trials]).astype(np.int32)
```

iii. Matches the reference.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Only `bp.hit` is used. Hit trials get outcome=1 (correct), and since ignore trials are already excluded, the remaining trials (misses) get outcome=0 (incorrect). Only two classes are produced instead of the reference's three (incorrect=0, correct=1, ignore=2).

ii.
```python
outcome = session['hit'][valid_trials].astype(np.int32)  # correct=1, incorrect=0
```

iii. The AI excludes ignore trials in the trial mask (`hit == 1 | miss == 1`), so only hit and miss remain.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A direct cast of `hit` to integer. No ignore class.

ii.
```python
outcome = session['hit'][valid_trials].astype(np.int32)
```

```python
'output_values': [
    ...
    ['incorrect', 'correct'],
    ...
]
```

iii. The reference keeps ignore trials and assigns them a third class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking data `obj.traj` from the side camera (cam 0) only, using the `tongue` feature. The AI uses only one camera view, whereas the reference uses both side (`tongue`) and bottom (`top_tongue`) views.

ii.
```python
tongue_vel = extract_kinematic_feature(session, 0, 'tongue', valid_trials, go_cue_times)
```

iii. The AI extracts tongue velocity from only the side camera. The reference combines both cameras to get better coverage since the tongue is visible in different frames in each view.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial: extract x,y positions from the tracking data, compute velocity via `np.gradient` with a fixed dt of 1/400 s, set velocity to 0 where tongue is not visible (NaN positions), align frame times to go cue using the video offset, then interpolate to the neural time axis using `interp1d`. Finally, NaN bins are filled with nearest-neighbor interpolation, and all-NaN trials are filled with 0. Discretization splits at the 50th percentile into two classes (0 and 1) with no "not visible" class.

ii.
```python
dx = np.gradient(x_pos, dt_vid)
dy = np.gradient(y_pos, dt_vid)
vel = np.sqrt(dx**2 + dy**2)
if is_tongue:
    nan_mask = np.isnan(x_pos) | np.isnan(y_pos)
    vel[nan_mask] = 0.0
# ...
interp_fn = interp1d(aligned_frame_times, vel, kind='linear', bounds_error=False, fill_value=np.nan)
velocities[:, ti] = interp_fn(TIME_AXIS)
```

```python
def discretize_per_session(values, percentile=50):
    valid = values[~np.isnan(values)]
    threshold = np.percentile(valid, percentile)
    discretized = (values >= threshold).astype(np.int32)
    return discretized
```

iii. Key differences from the reference: (1) No Gaussian smoothing of positions before differentiation. (2) Uses fixed dt_vid=1/400 for gradient rather than actual frame time differences. (3) Sets tongue velocity to 0 when not visible rather than NaN. (4) Interpolates to neural time axis rather than binning frame-by-frame. (5) Only two discretization classes (no "not visible" class). (6) Only one camera view.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Two categories only: 0 (below 50th percentile) and 1 (at or above 50th percentile). No "not visible" (class 2) category. Since tongue velocity is set to 0 when not visible, the median is likely 0, making nearly all bins class 1.

ii.
```python
tongue_vel_disc = discretize_per_session(tongue_vel)
# ...
'output_values': [..., ['low', 'high'], ...]
```

iii. The AI noted: "The tongue velocity being mostly 'high' is expected - when the tongue is not visible, velocity = 0, and median = 0, so everything >= 0 gets class 1. This is a property of the sparse tongue data."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by the video offset (`vidshift`) and go cue time, then the velocity is interpolated onto the neural time axis using `scipy.interpolate.interp1d`.

ii.
```python
aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
interp_fn = interp1d(aligned_frame_times, vel, kind='linear', bounds_error=False, fill_value=np.nan)
velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. The reference bins frames into the 5 ms grid by averaging, while the AI interpolates linearly. The video offset is computed using `nanmedian` rather than the reference's `mode`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Both `top_paw` and `bottom_paw` from the bottom camera (cam 1). The AI averages the two paw velocities when both are available. The reference uses only `top_paw`.

ii.
```python
paw_top = extract_kinematic_feature(session, 1, 'top_paw', valid_trials, go_cue_times)
paw_bot = extract_kinematic_feature(session, 1, 'bottom_paw', valid_trials, go_cue_times)
if paw_top is not None and paw_bot is not None:
    paw_vel = (paw_top + paw_bot) / 2.0
```

iii. The AI averages both paws, whereas the reference uses only `top_paw` because `bottom_paw` has unreliable tracking during the delay epoch.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as tongue velocity: extract x,y, compute velocity via gradient with fixed dt, fill missing with nearest-neighbor interpolation, interpolate to neural time axis, discretize at 50th percentile. No Gaussian smoothing of positions. No "not visible" class.

ii.
```python
# Same extract_kinematic_feature function as tongue, but is_tongue=False
if not is_tongue:
    for pos in [x_pos, y_pos]:
        mask = np.isnan(pos)
        if mask.any() and not mask.all():
            valid_idx = np.where(~mask)[0]
            pos[mask] = np.interp(np.where(mask)[0], valid_idx, pos[valid_idx])
```

iii. For non-tongue features, NaN positions are filled with nearest-neighbor interpolation before computing velocity, which fabricates positions where tracking failed.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Two categories: 0 (below 50th percentile) and 1 (at or above). No "not visible" class.

ii.
```python
paw_vel_disc = discretize_per_session(paw_vel)
```

iii. Same discretization as tongue velocity.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue velocity: video offset correction, then linear interpolation to neural time axis.

ii. Same `extract_kinematic_feature` function as tongue.

iii. Same approach.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The `motionEnergy_*.mat` file, loaded via `scipy.io.loadmat`. The AI extracts `me.data` and `me.moveThresh` from the file.

ii.
```python
def load_motion_energy(filepath):
    mat = scipy.io.loadmat(filepath, squeeze_me=False)
    me = mat['me']
    data_field = me['data'][0, 0]
    thresh_field = me['moveThresh'][0, 0]
    me_data = []
    for i in range(data_field.shape[0]):
        for j in range(data_field.shape[1]):
            trial_me = data_field[i, j].flatten().astype(float)
            me_data.append(trial_me)
    threshold = float(thresh_field.flatten()[0])
    return {'data': me_data, 'moveThresh': threshold}
```

iii. The AI loads both data and moveThresh, though moveThresh is not used for discretization (the 50th percentile is used instead). The reference handles the double-wrapped `me.data.data` case; the AI does not handle this edge case.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values are interpolated to the neural time axis using the side camera's frame times and the video offset. NaN bins are filled with nearest-neighbor interpolation; all-NaN trials are filled with 0. Discretized at the 50th percentile.

ii.
```python
def interpolate_motion_energy(me_data, session, valid_trials, go_cue_times):
    # ...
    aligned_frame_times = frame_times - vidshift - go_time
    interp_fn = interp1d(aligned_frame_times, me_trial,
                        kind='linear', bounds_error=False, fill_value=np.nan)
    me_interp[:, ti] = interp_fn(TIME_AXIS)
```

iii. The reference bins frames by averaging into 5 ms bins; the AI interpolates linearly.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Two categories: 0 (below 50th percentile) and 1 (at or above). No "no video" class.

ii.
```python
me_disc = discretize_per_session(me_interp)
```

iii. The instructions specify a third category (2: no video), which the AI omits.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Using the side camera's frame times corrected by video offset and go cue, then linearly interpolated to the neural time axis.

ii.
```python
if len(session['traj']) > 0 and trial_idx < len(session['traj'][0]['trials']):
    ft = session['traj'][0]['trials'][trial_idx].get('frameTimes')
# ...
aligned_frame_times = frame_times - vidshift - go_time
```

iii. The AI correctly uses the side camera's frame times for motion energy alignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI fills missing data rather than leaving it marked: (1) For tongue velocity, NaN positions result in velocity=0. (2) For paw velocity, NaN positions are filled with nearest-neighbor interpolation. (3) After interpolation to the neural time axis, remaining NaN bins are filled with nearest-neighbor interpolation, and all-NaN trials are filled with 0. (4) Sessions that fail to process are skipped with try/except. (5) Sessions without `clu` fields are skipped.

ii.
```python
# Fill NaN with nearest valid values (or 0 for tongue)
for ti in range(n_trials):
    col = velocities[:, ti]
    mask = np.isnan(col)
    if mask.any() and not mask.all():
        valid_idx = np.where(~mask)[0]
        col[mask] = np.interp(np.where(mask)[0], valid_idx, col[valid_idx])
    elif mask.all():
        velocities[:, ti] = 0.0
```

iii. The reference keeps NaN as NaN and maps it to a "not visible" class. The AI fills NaN with interpolated or zero values, then has only two discretization classes.

## 11-a. What are the most time-consuming steps of the code?

i. The per-trial, per-unit spike binning loop is likely the most time-consuming computational step, as it iterates over every unit and every trial individually. File loading is also expensive.

ii.
```python
for ui, ci in enumerate(cluster_indices):
    for ti, trial_idx in enumerate(valid_trials):
        # ... per-trial spike masking and histogram
        spike_mask = spike_trials == trial_num
        aligned_times = spike_times[spike_mask] - go_time
        counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The AI processes each unit-trial pair individually, which is much slower than the reference's vectorized `histogram2d` approach.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The double loop over units and trials for spike binning could be vectorized using `np.histogram2d` (as the reference does), processing all trials at once per unit. The per-trial smoothing loop in `causal_gaussian_smooth` could use `scipy.ndimage.gaussian_filter1d`. The per-trial kinematic feature extraction loop could potentially be parallelized.

ii.
```python
for ui, ci in enumerate(cluster_indices):
    unit = clusters[ci]
    for ti, trial_idx in enumerate(valid_trials):
        # ...
        counts, _ = np.histogram(aligned_times, bins=edges)
        fr_smooth = causal_gaussian_smooth(fr, smooth_samples)
        trialdat[:, ui, ti] = fr_smooth
```

iii. The nested unit x trial loop is the most obvious candidate for vectorization.

## 11-c. What processing does the code repeat multiple times?

i. The spike masking `spike_trials == trial_num` is computed for every trial within every unit's loop, re-scanning the entire spike array each time. The video offset computation uses `nanmedian` but is at least computed once per session. Frame time alignment is repeated per-trial across different kinematic features.

ii.
```python
for ti, trial_idx in enumerate(valid_trials):
    trial_num = trial_idx + 1
    spike_mask = spike_trials == trial_num  # recomputed for every trial
```

iii. The reference pre-indexes spikes by trial using `histogram2d`, avoiding repeated scanning.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `moveThresh` from motion energy files but never uses it (uses 50th percentile instead). It loads `sample` and `delay` event times that are never used. It reads `NdroppedFrames` from trajectory data but only uses it as a NaN check. It fills NaN velocities with interpolated values that then get discretized anyway.

ii.
```python
session['sample'] = ev['sample'][()].flatten() if 'sample' in ev else np.zeros(Ntrials)
session['delay'] = ev['delay'][()].flatten() if 'delay' in ev else np.zeros(Ntrials)
# ...
threshold = float(thresh_field.flatten()[0])  # loaded but never used for discretization
return {'data': me_data, 'moveThresh': threshold}
```

iii. These fields are loaded during the general session extraction but not used in the final conversion.
