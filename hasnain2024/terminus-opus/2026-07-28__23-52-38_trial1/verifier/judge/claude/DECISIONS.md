# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs the two data directories (`Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`) for files matching `data_structure_*.mat`, parses the loading scripts for probe info, and skips MAH (behavior-only) animals. It auto-detects whether each file is HDF5 (v7.3) or MATLAB v5, using separate loader functions. It finds 47 total session files.

ii.
```python
def get_available_sessions():
    for data_dir in ['data/Ephys_Behavior', 'data/RandomizedDelay_Ephys_Behavior']:
        for fn in sorted(os.listdir(data_dir)):
            if not fn.startswith('data_structure_'):
                continue
            ...
            if animal.startswith('MAH'):
                continue
            ...
            sessions.append({...})
    return sessions
```

```python
def load_session_data(filepath):
    fmt = detect_mat_format(filepath)
    if fmt == 'hdf5':
        return load_session_data_hdf5(filepath)
    elif fmt == 'v5':
        return load_session_data_v5(filepath)
```

iii. The AI chose to discover sessions by globbing data directories rather than hard-coding a session list. It parses the MATLAB loading scripts to determine which probe to use for each session. The CONVERSION_NOTES.md states "Found 47 total sessions".

## 1-b. How are the data split into subjects?

i. The animal ID is extracted from the filename pattern `data_structure_ANIMAL_DATE.mat` via regex. Unique subjects are collected as a sorted set at the end.

ii.
```python
m = re.match(r'data_structure_(\w+)_(\d{4}-\d{2}-\d{2})\.mat', fn)
animal = m.group(1)
...
subjects = sorted(set(r['animal'] for r in results))
subject_idx = np.array([subjects.index(r['animal']) for r in results])
```

iii. The AI extracts the animal name from the filename, which is the standard way this dataset identifies animals.

## 1-c. How are the data split into sessions?

i. Each `data_structure_*.mat` file is one session. The AI discovers sessions by globbing data directories and filters based on inclusion criteria (>40 R/L hit DR trials). It ends up with 43 sessions after excluding 2 for insufficient hit trials, 2 for missing clu data (no neural recordings), and including 1 session (JEB23_2023-10-20) not in the authors' loading scripts.

ii.
```python
for fn in sorted(os.listdir(data_dir)):
    if not fn.startswith('data_structure_'):
        continue
    ...
    sessions.append({...})
```

```python
def check_session_inclusion(session_data, probes):
    r_hit_dr = bp['R'] & bp['hit'] & ~bp['stim_enable'] & ~bp['autowater'] & ~bp['early']
    l_hit_dr = bp['L'] & bp['hit'] & ~bp['stim_enable'] & ~bp['autowater'] & ~bp['early']
    return n_r > MIN_HIT_TRIALS and n_l > MIN_HIT_TRIALS, n_r, n_l
```

iii. The AI applies the `UseInclusionCriteria.m` logic (>40 R and >40 L hit DR trials) as a session filter, which excludes some sessions that the reference includes.

## 1-d. How are the data split into trials?

i. Trials are indexed by Ntrials from `bp['Ntrials']`. Each behavioral array is read with that length. Spikes are binned per trial using trial numbers from `unit['trial']`.

ii.
```python
bp['Ntrials'] = int(get_field(bp_raw, 'Ntrials').flatten()[0])
...
for j in range(Ntrials):
    trial_num = j + 1  # MATLAB 1-indexed
    spk_mask = trial_nums == trial_num
    aligned_times = trialtm[spk_mask] - goCue[j]
    counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The AI uses the standard trial indexing from the Bpod table.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three filters: (1) early-lick trials (`bp.early`), (2) photostimulation trials (`bp.stim.enable`), and (3) **ignore trials** (`bp.no`) are all excluded. This means only hit and miss trials are kept. There is no filtering for trials past the recording end.

ii.
```python
valid_trials = ~bp['early'] & ~bp['stim_enable']
# Also exclude 'no' (ignore) trials
valid_trials = valid_trials & ~bp['no']
```

iii. The AI's CONVERSION_NOTES mentions excluding early and stim trials, following the paper. The ignore trial exclusion is done explicitly in the code but not clearly justified in the notes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from `obj.clu` spike-sorted clusters. Each unit has `trial` (which trial each spike belongs to), `trialtm` (spike time relative to trial start), and `quality` (curation label). Go cue times from `bp.ev.goCue` are used for alignment.

ii.
```python
unit['tm'] = unit_raw['tm'].flatten().astype(float)
unit['trial'] = unit_raw['trial'].flatten().astype(int)
unit['trialtm'] = unit_raw['trialtm'].flatten().astype(float)
unit['quality'] = ...
```

iii. Same raw variables as the reference.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 5ms time bins, converted to firing rates (Hz), and smoothed with a **causal** Gaussian kernel (`gausswin(N)` with the first half zeroed, N=15 bins = 75ms window). The smoothing is applied via convolution.

ii.
```python
def causal_gaussian_kernel(N):
    kern = np.array(sig_windows.gaussian(N, std=np.std(np.arange(1, N+1))))
    kern[:N//2] = 0
    kern = kern / kern.sum()
    return kern

def smooth_signal(x, N):
    kern = causal_gaussian_kernel(N)
    return np.convolve(x, kern, mode='same')

# In process_spikes:
fr = counts.astype(np.float64) / DT
fr_smooth = smooth_signal(fr, SMOOTH_N)
```

iii. The AI's CONVERSION_NOTES state: "Causal gaussian smoothing matching mySmooth.m". The smoothing matches the reference code's `mySmooth.m` which zeros the first half of the Gaussian window to make it causal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Clusters with quality containing 'garbage' are excluded. (2) Units with mean firing rate <= 0.5 Hz are excluded. Additionally, sessions with fewer than 10 units are excluded entirely.

ii.
```python
for u_idx, unit in enumerate(units):
    quality = unit['quality'].lower().strip()
    if 'garbage' not in quality:
        valid_units.append(u_idx)

# Low FR filter
mean_frs = np.mean(np.mean(firing_rates, axis=2), axis=1)
fr_mask = mean_frs > LOW_FR  # LOW_FR = 0.5
```

iii. The AI states it matches `deleteGarbageClu.m` which removes garbage clusters, and uses the code's `lowFR=0.5` threshold. The AI also applies a minimum units threshold: `MIN_UNITS = 10`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting `goCue[trial]` from `trialtm` for each spike, then binning into the time axis.

ii.
```python
aligned_times = trialtm[spk_mask] - goCue[j]
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. This matches the reference's `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 5ms (DT = 1/200). The time axis is created with `np.arange(TMIN, TMAX + DT/2, DT)` which produces **1001 bins** (from -2.5 to 2.5 inclusive), not 1000 as in the reference. No additional rebinning is applied.

ii.
```python
DT = 1.0 / 200.0  # 5ms time bins
TMIN = -2.5
TMAX = 2.5

def make_time_axis():
    time_axis = np.arange(TMIN, TMAX + DT/2, DT)
    time_axis = time_axis[time_axis <= TMAX + 1e-10]
    return time_axis
```

iii. The AI notes that TMIN=-2.5 and TMAX=2.5 match the reference code's `params.tmin` and `params.tmax`. However, the `arange` with `TMAX + DT/2` as the stop value produces 1001 bins instead of 1000.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself, computed from the parameters TMIN, TMAX, and DT. It is not derived from any raw data variable.

ii.
```python
time_axis = make_time_axis()
inp = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The time axis is a synthetic variable representing seconds from the go cue.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is created using `np.arange(TMIN, TMAX + DT/2, DT)`, producing 1001 evenly spaced values from -2.5 to 2.5. Each trial gets the same time axis as its input.

ii.
```python
time_axis = np.arange(TMIN, TMAX + DT/2, DT)
time_axis = time_axis[time_axis <= TMAX + 1e-10]
```

iii. No processing beyond computing the array.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis defines the bin edges for the neural data, so they share the same temporal grid by construction.

ii.
```python
edges = np.append(time_axis, time_axis[-1] + DT)
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. Alignment is implicit since both use the same time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `bp.R` (whether the instructed side is right). The AI uses the instructed side directly rather than inferring the actual lick direction from the combination of instructed side and outcome.

ii.
```python
lick_dir = 1 if bp['R'][trial_idx] else 0
```

iii. The AI maps R=1 (right) and L=0 (left) based on the instructed side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI directly uses the instructed side (`bp.R`) as lick direction, with right=1 and left=0. There are only 2 classes. Since ignore trials are excluded, there is no "no lick" class.

ii.
```python
lick_dir = 1 if bp['R'][trial_idx] else 0
```

```python
'output_values': [
    ['left', 'right'],           # lick_direction: 0=left, 1=right
    ...
]
```

iii. The AI equates lick direction with instructed side, which is only correct for hit trials. For miss trials, the animal licked the opposite direction. Since the AI excluded ignore trials, there is no "no lick" class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`. Autowater=True means WC context.

ii.
```python
context = 0 if bp['autowater'][trial_idx] else 1
```

iii. Same raw variable as the reference.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabelling: autowater trials are WC (0), all others are DR (1). Two classes.

ii.
```python
context = 0 if bp['autowater'][trial_idx] else 1
```

iii. Matches the reference encoding.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit`. Only hit/miss are represented since ignore trials are excluded.

ii.
```python
outcome = 1 if bp['hit'][trial_idx] else 0
```

iii. The AI uses only `hit` to determine outcome.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit trials are coded as correct (1), all others (which are miss trials, since ignores are excluded) are coded as incorrect (0). Only 2 classes.

ii.
```python
outcome = 1 if bp['hit'][trial_idx] else 0
```

```python
'output_values': [
    ...
    ['incorrect', 'correct'],    # outcome: 0=incorrect, 1=correct
    ...
]
```

iii. The instructions specify three classes: incorrect, correct, ignore. The AI drops ignore trials entirely.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from DLC tracking in `obj.traj`. The AI uses only the side camera (`cam_idx=0`) and the feature named `'tongue'`. Frame times and the video offset (from `sglx` bitcode) are also used.

ii.
```python
tongue_vel = extract_feature_velocity(session, 0, 'tongue', time_axis, vidshift)
```

iii. The AI uses only one camera view for the tongue, unlike the reference which combines both camera views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI (1) interpolates DLC x,y positions to the 5ms time axis using linear interpolation, (2) computes velocity via `np.gradient` on the interpolated positions, (3) sets NaN velocities to 0 for tongue, (4) computes speed as `sqrt(xvel^2 + yvel^2)`, and (5) discretizes at the 50th percentile with only 2 classes (low/high). NaN values are set to class 0 (low).

ii.
```python
f_x = interp1d(aligned_times[valid], x[valid], kind='linear',
              bounds_error=False, fill_value=np.nan)
xpos[:, trix] = f_x(time_axis)
...
xv = np.gradient(xpos[:, trix])
yv = np.gradient(ypos[:, trix])
xv[np.isnan(xv)] = 0
yv[np.isnan(yv)] = 0
...
vel_mag = np.sqrt(xvel**2 + yvel**2)
```

```python
tv_disc = np.zeros(len(time_axis), dtype=np.int64)
tv_disc[tv >= tongue_thresh] = 1
tv_disc[np.isnan(tv)] = 0
```

iii. The AI's approach differs from the reference in several ways: interpolating to the time axis before computing velocity (vs. computing velocity at frame resolution then binning), using only one camera view, and having only 2 discretization classes instead of 3.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Split at the 50th percentile of all valid trial data into 2 classes: 0 (below threshold / low) and 1 (above threshold / high). NaN values are set to 0 (low). There is no "not visible" class.

ii.
```python
tongue_thresh = np.nanpercentile(tongue_valid[~np.isnan(tongue_valid)], 50)
tv_disc = np.zeros(len(time_axis), dtype=np.int64)
tv_disc[tv >= tongue_thresh] = 1
tv_disc[np.isnan(tv)] = 0
```

iii. The instructions specify 3 classes (0: <50th pct, 1: >=50th pct, 2: not visible). The AI uses only 2.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by the video offset (`vidshift`) and aligned to the go cue, then the DLC positions are interpolated to the neural time axis using `interp1d`.

ii.
```python
aligned_times = frame_times - vidshift - goCue[trix]
f_x = interp1d(aligned_times[valid], x[valid], kind='linear',
              bounds_error=False, fill_value=np.nan)
xpos[:, trix] = f_x(time_axis)
```

iii. The AI uses the same video offset computation as the reference (matching `findVideoOffset.m`), but interpolates positions to the neural time axis rather than binning frame-resolution velocities.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from DLC tracking of `'top_paw'` from camera 1 (bottom camera, `cam_idx=1`).

ii.
```python
paw_vel = extract_feature_velocity(session, 1, 'top_paw', time_axis, vidshift)
```

iii. Uses the same feature (`top_paw`) as the reference, from the bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue velocity: (1) interpolate DLC positions to time axis, (2) compute velocity gradient, (3) subtract baseline derivative (median of diff), (4) fill NaN with nearest, (5) compute speed magnitude. Discretized at 50th percentile with 2 classes.

ii.
```python
# For non-tongue features:
base_x = np.nanmedian(np.diff(xpos[:, trix]))
base_y = np.nanmedian(np.diff(ypos[:, trix]))
xv = xv - base_x
yv = yv - base_y
xv = _fill_nearest(xv)
yv = _fill_nearest(yv)
```

```python
pv_disc = np.zeros(len(time_axis), dtype=np.int64)
pv_disc[pv >= paw_thresh] = 1
pv_disc[np.isnan(pv)] = 0
```

iii. The AI adds baseline subtraction for paw (matching `findVelocity.m` which subtracts baseline for non-tongue features) and nearest-fill for NaN values.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: split at 50th percentile into 2 classes. NaN set to 0. No "not visible" class.

ii.
```python
paw_thresh = np.nanpercentile(paw_valid[~np.isnan(paw_valid)], 50)
pv_disc[pv >= paw_thresh] = 1
pv_disc[np.isnan(pv)] = 0
```

iii. The instructions specify 3 classes (0, 1, 2: not visible). The AI uses only 2.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times corrected by video offset and go cue, then interpolated to neural time axis.

ii.
```python
aligned_times = frame_times - vidshift - goCue[trix]
f_x = interp1d(aligned_times[valid], x[valid], kind='linear', ...)
```

iii. Same alignment approach as tongue.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Derived from `motionEnergy_*.mat` files loaded via `scipy.io.loadmat`. The AI handles multiple format variants (struct with data/moveThresh fields, double-nested, or bare cell array).

ii.
```python
def load_motion_energy(me_filepath):
    data = sio.loadmat(me_filepath, squeeze_me=False)
    me_raw = data['me']
    if me_raw.dtype.names and 'data' in me_raw.dtype.names:
        me_data_field = me_raw['data'][0, 0]
        if hasattr(me_data_field, 'dtype') and me_data_field.dtype.names and 'data' in me_data_field.dtype.names:
            me_data_arr = me_data_field['data'][0, 0]
        else:
            me_data_arr = me_data_field
    else:
        me_data_arr = me_raw
```

iii. Matches the reference's handling of multiple motion energy file layouts.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is loaded, aligned to the go cue using frame times corrected by the video offset, and interpolated to the neural time axis using `interp1d`. NaN values are filled with nearest non-NaN value. Discretized at the 50th percentile with 2 classes.

ii.
```python
f_me = interp1d(aligned_times[:n], trial_me[:n], kind='linear',
               bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_me(time_axis)

# Fill NaN with nearest
for trix in range(Ntrials):
    if not np.all(np.isnan(me_aligned[:, trix])):
        me_aligned[:, trix] = _fill_nearest(me_aligned[:, trix])
```

iii. The AI interpolates and fills NaN values, unlike the reference which simply bins frame-resolution values.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Split at 50th percentile into 2 classes. NaN set to 0. No "no video" class.

ii.
```python
me_thresh = np.nanpercentile(me_valid[~np.isnan(me_valid)], 50)
me_disc = np.zeros(len(time_axis), dtype=np.int64)
me_disc[me >= me_thresh] = 1
me_disc[np.isnan(me)] = 0
```

iii. The instructions specify 3 classes (0, 1, 2: no video). The AI uses only 2.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from the side camera (camera 0) are corrected by video offset and go cue, then motion energy is interpolated to the neural time axis.

ii.
```python
cam_data = session['traj'][0]  # camera 0
trial_traj = cam_data['trials'][trix]
frame_times = trial_traj['frameTimes']
aligned_times = frame_times - vidshift - goCue[trix]
f_me = interp1d(aligned_times[:n], trial_me[:n], kind='linear', ...)
me_aligned[:, trix] = f_me(time_axis)
```

iii. Uses side camera frame times for alignment, which is correct since motion energy is computed from the side camera.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several cases: (1) sessions with missing `clu` data (JEB24_2023-10-03/04) are skipped via try/except, (2) trials with `NdroppedFrames` = NaN are skipped for velocity computation, (3) NaN positions are filled with nearest non-NaN values for paw velocity, (4) for tongue, NaN velocities are set to 0. However, the AI does **not** handle trials past the recording end, resulting in 30 all-zero neural data trials in sessions 35 and 42.

ii.
```python
if np.isnan(trial['NdroppedFrames']):
    continue
...
xv[np.isnan(xv)] = 0  # tongue
...
xv = _fill_nearest(xv)  # paw
```

iii. The all-zero neural data trials in sessions 35 and 42 are acknowledged in the CONVERSION_NOTES but not fixed.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the data files and spike processing. The AI's spike processing loops over every unit and every trial individually, making it significantly slower than the reference's vectorized approach.

ii.
```python
for i, u_idx in enumerate(valid_units):
    for j in range(Ntrials):
        spk_mask = trial_nums == trial_num
        aligned_times = trialtm[spk_mask] - goCue[j]
        counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. Total conversion time was ~5 minutes for 47 sessions.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner loop over trials in spike processing (lines 492-508) iterates over every trial for every unit, creating a boolean mask and calling `np.histogram` for each trial individually. The reference vectorizes this with a single `np.histogram2d` call per cluster. The velocity extraction also loops over trials.

ii.
```python
for j in range(Ntrials):
    trial_num = j + 1
    spk_mask = trial_nums == trial_num
    aligned_times = trialtm[spk_mask] - goCue[j]
    counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The double loop (units x trials) is the main bottleneck.

## 11-c. What processing does the code repeat multiple times?

i. The `_traj` access for each trial is done multiple times: once for the feature tracking and separately for frame times in motion energy alignment. The video offset is computed once per session, which is appropriate.

ii.
```python
# In extract_feature_velocity, traj data is accessed per trial
# In align_motion_energy, camera 0 traj data is accessed again per trial
cam_data = session['traj'][0]
trial_traj = cam_data['trials'][trix]
```

iii. The repeated trajectory access is minor since it's just dictionary lookups.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads many fields that are never used (e.g., `sample` and `delay` event times, `NdroppedFrames`). The AI also applies the `UseInclusionCriteria` session filter, which is not part of the decoder task and reduces the dataset. The `_fill_nearest` function is applied to motion energy NaN values, which are then discretized as class 0 anyway.

ii.
```python
bp['ev']['sample'] = get_field(ev_raw, 'sample').flatten().astype(float)
bp['ev']['delay'] = get_field(ev_raw, 'delay').flatten().astype(float)
bp['ev']['reward'] = ...
```

iii. Loading extra fields is minor overhead but the session inclusion filter reduces the dataset unnecessarily.
