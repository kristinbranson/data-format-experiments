# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by globbing the two data directories (`data/Ephys_Behavior` and `data/RandomizedDelay_Ephys_Behavior`) for files matching `data_structure_*.mat`, and parses the MATLAB loading scripts to determine probe assignments. It auto-detects whether each file is HDF5 (v7.3) or v5 format and uses different loaders accordingly. It finds 47 total data files.

ii.
```python
def get_available_sessions():
    for data_dir in ['data/Ephys_Behavior', 'data/RandomizedDelay_Ephys_Behavior']:
        for fn in sorted(os.listdir(data_dir)):
            if not fn.startswith('data_structure_'):
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

iii. CONVERSION_NOTES.md Step 1 documents parsing the loading scripts for probe info. The AI chose to dynamically discover sessions rather than hard-coding them.

## 1-b. How are the data split into subjects?

i. The animal name is extracted from the filename using a regex match (`data_structure_ANIMAL_DATE.mat`). Unique subjects are collected across all sessions and sorted.

ii.
```python
m = re.match(r'data_structure_(\w+)_(\d{4}-\d{2}-\d{2})\.mat', fn)
animal = m.group(1)
...
subjects = sorted(set(r['animal'] for r in results))
subject_idx = np.array([subjects.index(r['animal']) for r in results])
```

iii. The AI notes the animal ID is in the filename, consistent with the reference loading scripts.

## 1-c. How are the data split into sessions?

i. One session is one `data_structure_*.mat` file. The AI processes all discovered files (47 total), then applies inclusion criteria to filter down to 43 sessions. Sessions from both task folders (fixed-delay and randomized-delay) are treated uniformly.

ii.
```python
all_sessions = get_available_sessions()
print(f'Found {len(all_sessions)} total sessions')
...
for i, sess_info in enumerate(all_sessions):
    result = process_session(sess_info, time_axis, ...)
    if result is not None:
        results.append(result)
```

iii. The AI's CONVERSION_NOTES.md notes 25 Ephys_Behavior and 22 RandomizedDelay sessions on disk, yielding 47 total before filtering.

## 1-d. How are the data split into trials?

i. The number of trials is read from `bp.Ntrials`. Per-trial arrays (hit, miss, R, L, etc.) are read up to `Ntrials` entries. Each trial is one entry in these arrays.

ii.
```python
bp['Ntrials'] = int(f['obj/bp/Ntrials'][0, 0])
bp['hit'] = np.array(f['obj/bp/hit']).flatten().astype(bool)
...
```

iii. The trial structure follows from the Bpod behavioral data format.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) early-lick trials are excluded, (2) photostimulation trials are excluded, and (3) ignore trials (bp.no) are excluded. Additionally, sessions must pass inclusion criteria (>40 R hit DR and >40 L hit DR trials).

ii.
```python
valid_trials = ~bp['early'] & ~bp['stim_enable']
valid_trials = valid_trials & ~bp['no']
```

```python
def check_session_inclusion(session_data, probes):
    r_hit_dr = bp['R'] & bp['hit'] & ~bp['stim_enable'] & ~bp['autowater'] & ~bp['early']
    l_hit_dr = bp['L'] & bp['hit'] & ~bp['stim_enable'] & ~bp['autowater'] & ~bp['early']
    return n_r > MIN_HIT_TRIALS and n_l > MIN_HIT_TRIALS, n_r, n_l
```

iii. CONVERSION_NOTES.md Step 3 documents the session inclusion criteria from `UseInclusionCritera.m`. The AI also chose to exclude ignore trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Spike times are derived from `obj.clu` cluster data: `trial` (trial number), `trialtm` (spike time relative to trial start), and `quality` (cluster curation label). Go cue times from `bp.ev.goCue` provide the alignment event.

ii.
```python
unit['tm'] = unit_raw['tm'].flatten().astype(float)
unit['trial'] = unit_raw['trial'].flatten().astype(int)
unit['trialtm'] = unit_raw['trialtm'].flatten().astype(float)
unit['quality'] = ...
```

iii. Documented in CONVERSION_NOTES.md Step 1 under key functions.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue (`trialtm - goCue`), binned into 5ms bins using `np.histogram`, converted to firing rate (counts/dt), then smoothed with a causal Gaussian kernel. The causal kernel is built by taking `gausswin(15)`, zeroing the first half, and normalizing, matching the reference `mySmooth.m`. Smoothing is applied via `np.convolve(..., mode='same')`.

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

def process_spikes(session, probe_idx, time_axis):
    ...
    aligned_times = trialtm[spk_mask] - goCue[j]
    counts, _ = np.histogram(aligned_times, bins=edges)
    fr = counts.astype(np.float64) / DT
    fr_smooth = smooth_signal(fr, SMOOTH_N)
```

iii. CONVERSION_NOTES.md documents matching `mySmooth.m` and `getSeq.m` from the reference code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) clusters with 'garbage' in their quality label are removed; (2) units with mean firing rate <= 0.5 Hz are removed. Additionally, sessions with fewer than 10 units are excluded.

ii.
```python
for u_idx, unit in enumerate(units):
    quality = unit['quality'].lower().strip()
    if 'garbage' not in quality:
        valid_units.append(u_idx)

mean_frs = np.mean(np.mean(firing_rates, axis=2), axis=1)
fr_mask = mean_frs > LOW_FR  # LOW_FR = 0.5

if n_neurons < MIN_UNITS:  # MIN_UNITS = 10
    print(f'    EXCLUDED: only {n_neurons} units')
    return None
```

iii. CONVERSION_NOTES.md Step 1 notes `deleteGarbageClu.m` removes garbage clusters and `removeLowFRClusters.m` removes low FR units. Step 3 notes FR threshold of 0.5 Hz from the code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting `goCue[trial]` from `trialtm`, then binning into the time axis.

ii.
```python
aligned_times = trialtm[spk_mask] - goCue[j]
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. Documented as matching `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 5 ms (DT = 1/200), matching `params.dt` in the reference code. The time axis spans from -2.5 to 2.5 s, resulting in 1001 time bins. No rebinning is applied.

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

iii. CONVERSION_NOTES.md documents `dt=1/200` from `getDefaultParams.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This is the time axis itself, constructed from the parameters TMIN, TMAX, and DT. It is not derived from any raw data variable.

ii.
```python
time_axis = make_time_axis()
inp = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The time axis is defined to match the reference code parameters.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing. The time axis is generated as `np.arange(TMIN, TMAX + DT/2, DT)`, representing bin edges from -2.5 to 2.5 s at 5 ms intervals.

ii.
```python
time_axis = np.arange(TMIN, TMAX + DT/2, DT)
time_axis = time_axis[time_axis <= TMAX + 1e-10]
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same time axis is used for both the neural data binning and the input. Spikes are binned using histogram edges derived from this axis, and the input is the axis itself.

ii.
```python
edges = np.append(time_axis, time_axis[-1] + DT)
counts, _ = np.histogram(aligned_times, bins=edges)
...
inp = time_axis.astype(np.float32).reshape(1, -1)
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. `bp.R` (right-instructed trials). The lick direction is set directly from the instructed side.

ii.
```python
lick_dir = 1 if bp['R'][trial_idx] else 0
```

iii. CONVERSION_NOTES.md Step 5 maps `obj.bp.R` to `output[0]: lick_direction` with `R=1, L=0`.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Direct binary encoding: if `bp.R` is true, lick direction is 1 (right); otherwise 0 (left). Since ignore trials are excluded (see 1-e), there is no "no lick" class. The value is constant across all time bins for a given trial.

ii.
```python
lick_dir = 1 if bp['R'][trial_idx] else 0
out[0, :] = lick_dir  # constant across time
```

iii. The AI treats lick direction as the instructed direction rather than the actual lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater`. Autowater trials are classified as WC (water-cued, 0), all others as DR (delayed-response, 1).

ii.
```python
context = 0 if bp['autowater'][trial_idx] else 1
```

iii. Documented in CONVERSION_NOTES.md Step 5.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct binary encoding: autowater = WC (0), non-autowater = DR (1). Constant across time bins.

ii.
```python
context = 0 if bp['autowater'][trial_idx] else 1
out[1, :] = context
```

iii. Matches the instruction's WC=0, DR=1 coding.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit`. Hit trials are coded as correct (1), all others as incorrect (0).

ii.
```python
outcome = 1 if bp['hit'][trial_idx] else 0
```

iii. Documented in CONVERSION_NOTES.md Step 5.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Binary encoding: hit = correct (1), not hit = incorrect (0). Since ignore trials are already excluded in the trial filtering step, the remaining non-hit trials are misses. Constant across time bins.

ii.
```python
outcome = 1 if bp['hit'][trial_idx] else 0
out[2, :] = outcome
```

iii. The AI excludes ignore trials upstream, so the outcome is effectively binary.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DLC tracking data in `obj.traj`, specifically the `tongue` feature from camera 0 (side view). The AI extracts x, y coordinates and frame times from `traj.ts` and `traj.frameTimes`.

ii.
```python
tongue_vel = extract_feature_velocity(session, 0, 'tongue', time_axis, vidshift)
```

iii. CONVERSION_NOTES.md documents using `getKinematicsFromVideo.m` and `findVelocity.m` from the reference.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The x, y positions are extracted from `traj.ts` for frames where they are not NaN. Positions are interpolated to the neural time axis using linear interpolation. Then velocity magnitude is computed as `np.gradient` of x and y positions followed by `sqrt(xvel^2 + yvel^2)`. For tongue features, NaN velocities are set to 0 rather than filled with nearest neighbor. No smoothing is applied to positions before differentiation (unlike paw). The velocity is then discretized at the session 50th percentile.

ii.
```python
f_x = interp1d(aligned_times[valid], x[valid], kind='linear',
               bounds_error=False, fill_value=np.nan)
xpos[:, trix] = f_x(time_axis)
...
xv = np.gradient(xpos[:, trix])
...
xv[np.isnan(xv)] = 0
yv[np.isnan(yv)] = 0
...
vel_mag = np.sqrt(xvel**2 + yvel**2)
```

iii. CONVERSION_NOTES.md references `findVelocity.m` which computes velocity via gradient.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Discretized into 2 classes at the session 50th percentile: 0 = below threshold, 1 = above threshold. NaN values are set to 0 (below threshold).

ii.
```python
tongue_thresh = np.nanpercentile(tongue_valid[~np.isnan(tongue_valid)], 50)
tv_disc = np.zeros(len(time_axis), dtype=np.int64)
tv_disc[tv >= tongue_thresh] = 1
tv_disc[np.isnan(tv)] = 0
```

iii. Follows the instruction's 50th percentile discretization.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by the video offset (computed via `findVideoOffset.m` logic: `mode(bitcode.bitstart/fs) - mode(bitStart)`), then aligned to go cue. The DLC positions are interpolated to the neural time axis using `scipy.interpolate.interp1d`.

ii.
```python
aligned_times = frame_times - vidshift - goCue[trix]
f_x = interp1d(aligned_times[valid], x[valid], kind='linear',
               bounds_error=False, fill_value=np.nan)
xpos[:, trix] = f_x(time_axis)
```

iii. Video offset computation matches `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DLC tracking data in `obj.traj`, specifically the `top_paw` feature from camera 1 (bottom view).

ii.
```python
paw_vel = extract_feature_velocity(session, 1, 'top_paw', time_axis, vidshift)
```

iii. CONVERSION_NOTES.md documents using the bottom camera for paw tracking.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same `extract_feature_velocity` function as tongue. Positions are interpolated to the neural time axis, NaN values are filled with nearest neighbor, velocity is computed via `np.gradient`, baseline derivative is subtracted (`np.nanmedian(np.diff(x))` and similar for y), and then velocity magnitude is computed. NaN velocity values are filled with nearest neighbor.

ii.
```python
if not is_tongue:
    mask_nan = np.isnan(xpos[:, trix])
    if np.any(mask_nan) and not np.all(mask_nan):
        xpos[:, trix] = _fill_nearest(xpos[:, trix])
        ypos[:, trix] = _fill_nearest(ypos[:, trix])
...
if not is_tongue:
    base_x = np.nanmedian(np.diff(xpos[:, trix]))
    base_y = np.nanmedian(np.diff(ypos[:, trix]))
    xv = xv - base_x
    yv = yv - base_y
    xv = _fill_nearest(xv)
    yv = _fill_nearest(yv)
```

iii. The baseline subtraction for non-tongue features attempts to match `findVelocity.m`.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Discretized into 2 classes at the session 50th percentile: 0 = below threshold, 1 = above threshold. NaN values are set to 0.

ii.
```python
paw_thresh = np.nanpercentile(paw_valid[~np.isnan(paw_valid)], 50)
pv_disc = np.zeros(len(time_axis), dtype=np.int64)
pv_disc[pv >= paw_thresh] = 1
pv_disc[np.isnan(pv)] = 0
```

iii. Follows the instruction's 50th percentile discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times corrected by video offset, aligned to go cue, then positions interpolated to neural time axis using `interp1d`.

ii.
```python
aligned_times = frame_times - vidshift - goCue[trix]
f_x = interp1d(aligned_times[valid], x[valid], kind='linear', ...)
```

iii. Same video offset and interpolation approach as tongue velocity.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Separate `motionEnergy_*.mat` files loaded with `scipy.io.loadmat`. Handles multiple format variants (direct array, struct with data/moveThresh, double-nested struct).

ii.
```python
def load_motion_energy(me_filepath):
    data = sio.loadmat(me_filepath, squeeze_me=False)
    me_raw = data['me']
    # Handles multiple format variants...
    return {'data': me_trials, 'moveThresh': me_thresh}
```

iii. CONVERSION_NOTES.md Step 1 documents `loadMotionEnergy.m`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy per-trial values are interpolated to the neural time axis using `interp1d`. Frame times come from the side camera (camera 0). NaN values are filled with nearest neighbor. Then discretized at the session 50th percentile.

ii.
```python
f_me = interp1d(aligned_times[:n], trial_me[:n], kind='linear',
                bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_me(time_axis)
...
for trix in range(Ntrials):
    if not np.all(np.isnan(me_aligned[:, trix])):
        me_aligned[:, trix] = _fill_nearest(me_aligned[:, trix])
```

iii. NaN filling with nearest neighbor is applied after interpolation.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Discretized into 2 classes at the session 50th percentile: 0 = below threshold, 1 = above threshold. NaN values set to 0.

ii.
```python
me_thresh = np.nanpercentile(me_valid[~np.isnan(me_valid)], 50)
me_disc = np.zeros(len(time_axis), dtype=np.int64)
me_disc[me >= me_thresh] = 1
me_disc[np.isnan(me)] = 0
```

iii. Follows instruction's 50th percentile discretization.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from the side camera are corrected by video offset and aligned to go cue, then motion energy values are interpolated to the neural time axis.

ii.
```python
frame_times = trial_traj['frameTimes']
aligned_times = frame_times - vidshift - goCue[trix]
f_me = interp1d(aligned_times[:n], trial_me[:n], kind='linear', ...)
me_aligned[:, trix] = f_me(time_axis)
```

iii. Same video offset computation as other camera-based outputs.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches: (1) Sessions without `clu` data (JEB24_2023-10-03/04) are skipped with an error. (2) Missing frame times trigger a fallback: frame times are generated from frame count and VIDEO_FS (400 Hz). (3) NaN positions are excluded from interpolation. (4) After interpolation, NaN values for non-tongue features are filled with nearest neighbor. (5) Tongue NaN velocities are set to 0. (6) Trials with NaN NdroppedFrames are skipped for video processing.

ii.
```python
if trial['frameTimes'] is not None and not np.all(np.isnan(trial['frameTimes'])):
    frame_times = trial['frameTimes']
else:
    n_frames = ts.shape[2]
    frame_times = np.arange(1, n_frames + 1) / VIDEO_FS
...
if not is_tongue:
    xpos[:, trix] = _fill_nearest(xpos[:, trix])
...
xv[np.isnan(xv)] = 0  # for tongue
```

iii. CONVERSION_NOTES.md Step 10 documents handling of probes with invalid clu data and format differences.

## 11-a. What are the most time-consuming steps of the code?

i. The spike processing loop is the most time-consuming, iterating over every unit and every trial individually. For each unit and each trial, it masks spikes, aligns to goCue, bins with histogram, and smooths. Loading files is also slow. Total conversion takes about 5 minutes for all sessions.

ii.
```python
for i, u_idx in enumerate(valid_units):
    unit = units[u_idx]
    for j in range(Ntrials):
        trial_num = j + 1
        spk_mask = trial_nums == trial_num
        aligned_times = trialtm[spk_mask] - goCue[j]
        counts, _ = np.histogram(aligned_times, bins=edges)
        fr = counts.astype(np.float64) / DT
        fr_smooth = smooth_signal(fr, SMOOTH_N)
        firing_rates[i, :, j] = fr_smooth
```

iii. CONVERSION_NOTES.md Step 7 estimates ~8s per session.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The main inefficiency is the nested loop over units and trials in `process_spikes`. The reference solution uses `np.histogram2d` to count all spikes for a unit across all trials in a single call, which is significantly faster. The AI's code loops trial-by-trial for each unit.

ii.
```python
# AI's per-trial loop:
for i, u_idx in enumerate(valid_units):
    for j in range(Ntrials):
        spk_mask = trial_nums == trial_num
        counts, _ = np.histogram(aligned_times, bins=edges)

# Reference's vectorized approach:
counts, _, _ = np.histogram2d(spike_trial, spike_time, bins=[trial_edges, BIN_EDGES])
```

iii. No explicit justification given for the loop-based approach.

## 11-c. What processing does the code repeat multiple times?

i. The `_traj` access is repeated when extracting different features from the same trial. Frame times are loaded redundantly when processing tongue and then paw from overlapping cameras. The `extract_feature_velocity` function is called separately for tongue and paw, each re-reading frame times for every trial.

ii.
```python
tongue_vel = extract_feature_velocity(session, 0, 'tongue', time_axis, vidshift)
paw_vel = extract_feature_velocity(session, 1, 'top_paw', time_axis, vidshift)
```

iii. No explicit justification for the repeated access.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads many fields from the data structure that are never used (e.g., `ev.sample`, `ev.delay`, `ev.reward`, `NdroppedFrames` for some uses). The `moveThresh` from motion energy is loaded but not used. The AI also computes the `detect_mat_format` check which reads the file header separately before loading, adding an extra file read.

ii.
```python
bp['ev']['sample'] = get_field(ev_raw, 'sample').flatten().astype(float)
bp['ev']['delay'] = get_field(ev_raw, 'delay').flatten().astype(float)
bp['ev']['reward'] = get_field(ev_raw, 'reward').flatten().astype(float)
...
me_thresh = float(me_raw['moveThresh'][0, 0].flatten()[0])
```

iii. No explicit justification; these appear to be loaded "just in case."
