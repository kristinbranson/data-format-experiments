# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans two data directories (`data/Ephys_Behavior` and `data/RandomizedDelay_Ephys_Behavior`) for files matching the pattern `data_structure_ANIMAL_DATE.mat`. It skips MAH-prefixed animals (behavior-only). It parses MATLAB loading scripts in `code/DataLoadingScripts/Recording and video/` to determine which probes to use for each session. It auto-detects HDF5 vs MATLAB v5 format and loads behavioral data (`obj.bp`), cluster/spike data (`obj.clu`), trajectory data (`obj.traj`), and SpikeGLX metadata (`obj.sglx`). Motion energy is loaded from separate `motionEnergy_ANIMAL_DATE.mat` files.

ii.
```python
def get_available_sessions():
    for data_dir in ['data/Ephys_Behavior', 'data/RandomizedDelay_Ephys_Behavior']:
        for fn in sorted(os.listdir(data_dir)):
            if not fn.startswith('data_structure_'):
                continue
            m = re.match(r'data_structure_(\w+)_(\d{4}-\d{2}-\d{2})\.mat', fn)
            ...
            if animal.startswith('MAH'):
                continue
            ...

def load_session_data(filepath):
    fmt = detect_mat_format(filepath)
    if fmt == 'hdf5':
        return load_session_data_hdf5(filepath)
    elif fmt == 'v5':
        return load_session_data_v5(filepath)
```

iii. The AI documented in CONVERSION_NOTES.md that it identified the data directories and file formats. It correctly skips MAH animals (behavior-only optogenetic sessions) and handles dual .mat formats.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the animal name extracted from data filenames (`data_structure_ANIMAL_DATE.mat`). All unique animal names from included sessions form the subjects list.

ii.
```python
m = re.match(r'data_structure_(\w+)_(\d{4}-\d{2}-\d{2})\.mat', fn)
animal = m.group(1)
...
subjects = sorted(set(r['animal'] for r in results))
subject_idx = np.array([subjects.index(r['animal']) for r in results])
```

iii. The AI noted 14 total animals across both datasets. The paper reports 9 DR mice and 4 RD mice. The AI included all available animals from the data files.

## 1-c. How are the data split into sessions?

i. Each `.mat` file corresponds to one session. Sessions are identified by the combination of animal name and date. Multi-probe sessions have probes concatenated into a single session. Sessions are filtered by inclusion criteria (>40 R hit DR trials AND >40 L hit DR trials) and minimum unit count (>=10 units after quality and FR filtering).

ii.
```python
sessions.append({
    'animal': animal,
    'date': date,
    'filepath': filepath,
    'probes': probes,
    ...
})
...
included, n_r, n_l = check_session_inclusion(session, probes)
if not included:
    return None
...
if n_neurons < MIN_UNITS:
    return None
```

iii. The AI documented that it uses the same inclusion criteria as `UseInclusionCritera.m` and the minimum unit threshold from the reference code.

## 1-d. How are the data split into trials?

i. Trials are defined by the `obj.bp.Ntrials` field in each session's data file. Each trial has associated behavioral flags (L, R, hit, miss, no, autowater, early, stim_enable) and event times (goCue, sample, delay, reward).

ii.
```python
bp['Ntrials'] = int(f['obj/bp/Ntrials'][0, 0])
...
for j in range(Ntrials):
    trial_num = j + 1  # MATLAB 1-indexed
```

iii. The AI correctly reads the trial count from the behavioral data structure and iterates over all trials during spike processing.

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes trials that are: early lick (`early`), photostimulation (`stim_enable`), or ignore/no-response (`no`). Remaining trials include both hit and miss trials across both DR and WC contexts.

ii.
```python
valid_trials = ~bp['early'] & ~bp['stim_enable']
valid_trials = valid_trials & ~bp['no']
trial_indices = np.where(valid_trials)[0]
```

iii. The AI noted this matches the paper's statement that "early lick and ignore trials were omitted from all analyses." Including miss trials is necessary for the outcome decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times stored in `obj.clu`, specifically `clu.trialtm` (spike times relative to trial start) and `clu.trial` (which trial each spike belongs to), aligned to the go cue using `obj.bp.ev.goCue`.

ii.
```python
unit['tm'] = np.array(f[clu_group['tm'][u, 0]]).flatten()
unit['trial'] = np.array(f[clu_group['trial'][u, 0]]).flatten().astype(int)
unit['trialtm'] = np.array(f[clu_group['trialtm'][u, 0]]).flatten()
```

iii. The AI documented that spike data comes from `obj.clu` and is aligned to the goCue event, matching `alignSpikes.m`.

## 2-b. How is the `neural` data processed?

i. Processing pipeline: (1) Remove garbage-quality clusters. (2) For each remaining unit and each trial, align spike times to goCue: `trialtm - goCue[trial]`. (3) Bin aligned spikes into 5ms time bins using `np.histogram`. (4) Convert spike counts to firing rate by dividing by dt (0.005s). (5) Smooth with a causal Gaussian kernel (window=15 bins). (6) Remove low-firing-rate units (mean FR < 0.5 Hz). (7) Concatenate units across probes.

ii.
```python
aligned_times = trialtm[spk_mask] - goCue[j]
counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts.astype(np.float64) / DT
fr_smooth = smooth_signal(fr, SMOOTH_N)
...
mean_frs = np.mean(np.mean(firing_rates, axis=2), axis=1)
fr_mask = mean_frs > LOW_FR
```

iii. The AI documented that this matches the reference pipeline: `deleteGarbageClu` -> `alignSpikes` -> `getSeq` (bin + smooth) -> `removeLowFRClusters`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filtering steps: (1) Clusters with quality label containing 'garbage' are excluded. (2) Units with mean firing rate <= 0.5 Hz across all trials and time bins are excluded. Sessions with fewer than 10 remaining units are excluded entirely.

ii.
```python
# Quality filter
quality = unit['quality'].lower().strip()
if 'garbage' not in quality:
    valid_units.append(u_idx)
...
# FR filter
mean_frs = np.mean(np.mean(firing_rates, axis=2), axis=1)
fr_mask = mean_frs > LOW_FR
...
# Session filter
if n_neurons < MIN_UNITS:
    return None
```

iii. The AI referenced `deleteGarbageClu.m` and `removeLowFRClusters.m` as the basis for these filters. The MIN_UNITS=10 threshold is from the paper: "Recording sessions were included for analysis only if they had at least 10 units."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting the go cue time for each trial: `aligned_time = trialtm - goCue[trial]`. This matches `alignSpikes.m` which computes `trialtm_aligned = trialtm - event`.

ii.
```python
aligned_times = trialtm[spk_mask] - goCue[j]
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The AI set `ALIGN_EVENT = 'goCue'` matching `params.alignEvent = 'goCue'` in the reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 5 ms (dt = 1/200 s), matching `params.dt = 1/200` in the reference code. The time window is [-2.5, 2.5] seconds relative to go cue. The AI produces 1001 time bins per trial.

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

iii. The AI documented that the 5ms bin size and [-2.5, 2.5] time range match `getDefaultParams.m`. No temporal rebinning is applied beyond the initial 5ms binning. However, the reference code produces 1000 time bins (using `histc` with edges and removing the last bin), while the AI produces 1001 bins.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not derived from any raw variable. It is the time axis itself: evenly spaced values from -2.5 to 2.5 seconds, representing time relative to go cue onset. It is the same for every trial.

ii.
```python
inp = time_axis.astype(np.float32).reshape(1, -1)  # (1, n_timepoints)
input_trials.append(inp)
```

iii. The AI noted that the time axis is constructed from the parameters (TMIN, TMAX, DT) and is identical for all trials within and across sessions.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing is involved. The time axis is directly constructed as `np.arange(-2.5, 2.5+dt/2, dt)` and used as the input for every trial.

ii.
```python
time_axis = np.arange(TMIN, TMAX + DT/2, DT)
...
inp = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The AI correctly identified this as a simple time vector that is constant across trials.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is the same array used to define the neural data binning edges, so it is inherently aligned. Both neural spike histograms and the input time vector use the same `time_axis`.

ii.
```python
edges = np.append(time_axis, time_axis[-1] + DT)
...
counts, _ = np.histogram(aligned_times, bins=edges)
...
inp = time_axis.astype(np.float32).reshape(1, -1)
```

iii. Alignment is ensured by construction since both neural and input data share the same time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R` (right lick trials) and `obj.bp.L` (left lick trials), which are boolean arrays indicating the lick direction for each trial.

ii.
```python
bp['R'] = np.array(f['obj/bp/R']).flatten().astype(bool)
bp['L'] = np.array(f['obj/bp/L']).flatten().astype(bool)
...
lick_dir = 1 if bp['R'][trial_idx] else 0
```

iii. The AI mapped right licks to 1 and left licks to 0, matching the instruction specification: "left = 0, right = 1."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Simple binary mapping: if `bp.R` is True for the trial, lick_direction = 1 (right); otherwise lick_direction = 0 (left). This is a per-trial constant broadcast across all time bins.

ii.
```python
lick_dir = 1 if bp['R'][trial_idx] else 0
...
out[0, :] = lick_dir  # constant across time
```

iii. The AI treated this as a per-trial scalar broadcast to all time points, consistent with the instruction that it can be per-trial.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`, a boolean array indicating whether each trial was in the water-cued (WC) context (autowater=True) or delayed-response (DR) context (autowater=False).

ii.
```python
bp['autowater'] = np.array(f['obj/bp/autowater']).flatten().astype(bool)
...
context = 0 if bp['autowater'][trial_idx] else 1
```

iii. The AI identified `autowater` as the WC flag based on the reference code's condition strings, e.g., `'R&hit&~stim.enable&autowater&~early'` for WC trials.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Simple binary mapping: if `autowater` is True, context = 0 (WC); if False, context = 1 (DR). Broadcast across all time bins as a per-trial constant.

ii.
```python
context = 0 if bp['autowater'][trial_idx] else 1
out[1, :] = context
```

iii. The AI mapped WC=0 and DR=1 matching the instruction specification.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`, a boolean array indicating whether each trial was correct (hit=True) or incorrect (hit=False, including miss and ignore).

ii.
```python
bp['hit'] = np.array(f['obj/bp/hit']).flatten().astype(bool)
...
outcome = 1 if bp['hit'][trial_idx] else 0
```

iii. The AI used `hit` to determine correct trials, consistent with the instruction "incorrect = 0, correct = 1."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Simple binary mapping: hit=True maps to 1 (correct), hit=False maps to 0 (incorrect). Broadcast across all time bins.

ii.
```python
outcome = 1 if bp['hit'][trial_idx] else 0
out[2, :] = outcome
```

iii. Since ignore (`no`) trials are already filtered out in trial selection, the remaining incorrect trials are miss trials.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj{1}` (camera 0, side view), specifically the trajectory data for the feature named 'tongue'. The x and y positions are extracted from `traj.ts` (DLC tracking data) and `traj.frameTimes` (video frame timestamps).

ii.
```python
tongue_vel = extract_feature_velocity(session, 0, 'tongue', time_axis, vidshift)
...
x = ts[feat_idx, 0, :]  # x coordinate
y = ts[feat_idx, 1, :]  # y coordinate
```

iii. The AI correctly identified tongue tracking from camera 0, matching the reference code's `params.traj_features{1}` which includes 'tongue' for view 1 (camera 0).

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Processing steps: (1) Extract x, y positions from DLC tracking. (2) Align frame times to goCue using video offset: `aligned_times = frameTimes - vidshift - goCue[trial]`. (3) Interpolate x, y positions to the neural time axis. (4) Compute velocity as `np.gradient()` of interpolated positions. (5) For tongue, set NaN velocities to 0 (tongue not visible). (6) Compute velocity magnitude: `sqrt(xvel^2 + yvel^2)`.

ii.
```python
aligned_times = frame_times - vidshift - goCue[trix]
f_x = interp1d(aligned_times[valid], x[valid], kind='linear', bounds_error=False, fill_value=np.nan)
xpos[:, trix] = f_x(time_axis)
...
xv = np.gradient(xpos[:, trix])
xv[np.isnan(xv)] = 0  # tongue: set to 0 if not visible
...
vel_mag = np.sqrt(xvel**2 + yvel**2)
```

iii. The AI documented matching `findPosition.m` and `findVelocity.m` for the processing pipeline. However, the reference code fills tongue NaN positions with a mean baseline position (via `setTongueBaselinePosition`) before computing velocity, while the AI skips this step and directly sets NaN velocities to 0.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The 50th percentile of tongue velocity magnitude across all valid trials and time points within a session is used as the threshold. Values >= threshold are "high" (1), values < threshold are "low" (0). NaN values are set to 0 (low).

ii.
```python
tongue_valid = tongue_vel[:, trial_indices]
tongue_thresh = np.nanpercentile(tongue_valid[~np.isnan(tongue_valid)], 50)
...
tv_disc = np.zeros(len(time_axis), dtype=np.int64)
tv_disc[tv >= tongue_thresh] = 1
tv_disc[np.isnan(tv)] = 0
```

iii. The AI followed the instruction specification for 50th percentile per-session thresholding. However, many sessions have a tongue velocity threshold of 0.00 (because the tongue is invisible most of the time, producing velocity magnitude of 0), resulting in nearly 100% "high" classification.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue position is interpolated from video frame times (400 Hz) to the neural time axis (5ms bins) using linear interpolation, after aligning frame times to goCue with the video offset correction. Velocity is then computed on the interpolated time axis.

ii.
```python
f_x = interp1d(aligned_times[valid], x[valid], kind='linear', bounds_error=False, fill_value=np.nan)
xpos[:, trix] = f_x(time_axis)
```

iii. This matches the reference `findPosition.m` which uses `interp1(frameTimes-vidshift-alignEvent, ts, taxis)`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj{2}` (camera 1, bottom view), specifically the feature named 'top_paw'. The x and y positions come from DLC tracking data.

ii.
```python
paw_vel = extract_feature_velocity(session, 1, 'top_paw', time_axis, vidshift)
```

iii. The AI correctly identified `top_paw` from camera 1 (bottom view), consistent with the reference code's `params.traj_features{2}` which includes 'top_paw'.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as tongue but with paw-specific handling: (1) Extract x, y positions. (2) Align and interpolate to neural time axis. (3) Fill missing values with nearest for non-tongue features. (4) Compute velocity via `np.gradient()`. (5) Subtract baseline derivative (median of diff) for non-tongue features. (6) Fill velocity NaN with nearest. (7) Compute velocity magnitude.

ii.
```python
# Non-tongue: fill NaN positions
if not is_tongue:
    xpos[:, trix] = _fill_nearest(xpos[:, trix])
    ypos[:, trix] = _fill_nearest(ypos[:, trix])
...
# Subtract baseline
base_x = np.nanmedian(np.diff(xpos[:, trix]))
base_y = np.nanmedian(np.diff(ypos[:, trix]))
xv = xv - base_x
yv = yv - base_y
```

iii. The AI documented matching `findPosition.m` (fillmissing nearest) and `findVelocity.m` (baseline subtraction). However, the reference code has a bug where `basederiv(1)` (x baseline) is subtracted from both x and y velocities. The AI "fixed" this by using separate x and y baselines, which deviates from the reference.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same approach as tongue velocity: 50th percentile per-session threshold on paw velocity magnitude. Values >= threshold = 1 (high), < threshold = 0 (low).

ii.
```python
paw_valid = paw_vel[:, trial_indices]
paw_thresh = np.nanpercentile(paw_valid[~np.isnan(paw_valid)], 50)
...
pv_disc = np.zeros(len(time_axis), dtype=np.int64)
pv_disc[pv >= paw_thresh] = 1
```

iii. Follows instruction specification for 50th percentile thresholding.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment approach as tongue: interpolation from video frame times to neural time axis, with video offset correction relative to goCue.

ii.
```python
aligned_times = frame_times - vidshift - goCue[trix]
f_x = interp1d(aligned_times[valid], x[valid], kind='linear', bounds_error=False, fill_value=np.nan)
xpos[:, trix] = f_x(time_axis)
```

iii. Matches reference `findPosition.m`.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate `motionEnergy_ANIMAL_DATE.mat` files. The variable `me.data` contains per-trial motion energy time series, and `me.moveThresh` contains the per-session movement threshold (not used for discretization).

ii.
```python
me_fn = fn.replace('data_structure_', 'motionEnergy_')
me_filepath = os.path.join(data_dir, me_fn)
...
data = sio.loadmat(me_filepath, squeeze_me=False)
me_raw = data['me']
```

iii. The AI correctly identified the motion energy files and their naming convention.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Processing: (1) Load motion energy from .mat file, handling nested struct format (`me.data.data` pattern). (2) Align to goCue using video frame times and video offset: `interp1(frameTimes - vidshift - goCue, me_data, time_axis)`. (3) Fill NaN values with nearest non-NaN value.

ii.
```python
def align_motion_energy(me_data, session, time_axis, vidshift):
    for trix in range(min(Ntrials, n_me_trials)):
        frame_times = trial_traj['frameTimes']
        aligned_times = frame_times - vidshift - goCue[trix]
        f_me = interp1d(aligned_times[:n], trial_me[:n], kind='linear',
                       bounds_error=False, fill_value=np.nan)
        me_aligned[:, trix] = f_me(time_axis)
    ...
    me_aligned[:, trix] = _fill_nearest(me_aligned[:, trix])
```

iii. The AI documented matching `loadMotionEnergy.m` which uses `interp1(frameTimes-vidshift-alignTimes, me.data, taxis)` and `fillmissing(me.data, 'nearest')`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. 50th percentile per-session threshold on motion energy across valid trials and time points. Values >= threshold = 1 (high), < threshold = 0 (low).

ii.
```python
me_valid = me_aligned[:, trial_indices]
me_thresh = np.nanpercentile(me_valid[~np.isnan(me_valid)], 50)
...
me_disc = np.zeros(len(time_axis), dtype=np.int64)
me_disc[me >= me_thresh] = 1
```

iii. Follows instruction specification. The reference code uses a manually-defined `moveThresh`, but the instructions explicitly specify 50th percentile thresholding for the decoder.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is interpolated from video frame times (400 Hz) to the neural time axis using linear interpolation, after aligning frame times to goCue with video offset correction. This matches the approach used for tongue and paw velocities.

ii.
```python
aligned_times = frame_times - vidshift - goCue[trix]
f_me = interp1d(aligned_times[:n], trial_me[:n], kind='linear',
               bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_me(time_axis)
```

iii. Matches `loadMotionEnergy.m`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several types of missing/problematic data are handled: (1) NaN event times: trials with NaN goCue are still processed (spikes with NaN alignment produce no histogram counts). (2) Missing video data: trials with NaN frameTimes or NaN NdroppedFrames are skipped for trajectory extraction. (3) Missing motion energy files: filled with NaN, then NaN set to 0 during discretization. (4) Missing tongue visibility: NaN tongue velocity set to 0. (5) Missing paw/non-tongue positions: filled with nearest non-NaN value. (6) Probes with no valid clu data: skipped. (7) Sessions with no valid units or insufficient trials: excluded. (8) Dual .mat format (v5 vs v7.3): auto-detected.

ii.
```python
if trial['ts'] is None:
    continue
if np.isnan(trial['NdroppedFrames']):
    continue
...
xv[np.isnan(xv)] = 0  # tongue
...
xpos[:, trix] = _fill_nearest(xpos[:, trix])  # non-tongue
...
me_aligned[:, trix] = _fill_nearest(me_aligned[:, trix])
```

iii. The AI documented handling of edge cases including probes with invalid data (JEB6 probe 1), sessions without clu data (JEB24), and format differences.

## 11-a. What are the most time-consuming steps of the code?

i. Based on the conversion output, the most time-consuming steps are: (1) Loading .mat files (2-5s per session). (2) Spike processing including binning and smoothing (1-6s per session, scaling with number of units). (3) Paw velocity extraction (0.3-1.0s per session). Total conversion time for 47 sessions was approximately 5-7 minutes.

ii.
```python
# Timing in process_session:
t0 = time.time()
session = load_session_data(sess_info['filepath'])
t_load = time.time() - t0
...
fr, kept = process_spikes(session, p_idx, time_axis)
t_spk = time.time() - t0
```

iii. The AI estimated ~8s per session and ~5 minutes total, and noted this was within the 15-minute acceptable threshold.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several nested loops could be vectorized: (1) The spike processing loop iterates over units and trials individually (`for i, u_idx in enumerate(valid_units): for j in range(Ntrials)`). This could be vectorized by using sparse matrices or batch histogram operations. (2) The `_fill_nearest` function uses an inner loop over array indices. (3) The velocity computation loops over trials (`for trix in range(Ntrials)`).

ii.
```python
for i, u_idx in enumerate(valid_units):
    for j in range(Ntrials):
        spk_mask = trial_nums == trial_num
        aligned_times = trialtm[spk_mask] - goCue[j]
        counts, _ = np.histogram(aligned_times, bins=edges)
        fr = counts.astype(np.float64) / DT
        fr_smooth = smooth_signal(fr, SMOOTH_N)
        firing_rates[i, :, j] = fr_smooth
```

iii. The AI acknowledged this in Step 7 estimates but did not optimize further since the total runtime was acceptable.

## 11-c. What processing does the code repeat multiple times?

i. (1) The video offset (`compute_video_offset`) is called once per session in `process_session`, but it is also implicitly used in both `extract_feature_velocity` and `align_motion_energy`, so it is computed once and passed. (2) The smoothing kernel (`causal_gaussian_kernel`) is recomputed for every call to `smooth_signal`, which happens for every unit on every trial. (3) The time axis and edges are recomputed or re-derived in multiple functions.

ii.
```python
def smooth_signal(x, N):
    kern = causal_gaussian_kernel(N)  # recomputed every call
    return np.convolve(x, kern, mode='same')
```

iii. The smoothing kernel recomputation is wasteful since N is constant (15), but the overhead is small relative to the convolution.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The code loads ALL trajectory features and camera data, but only uses 'tongue' from camera 0 and 'top_paw' from camera 1. Other features (jaw, nose, trident, etc.) are loaded but never used. (2) The code loads the motion energy `moveThresh` field but never uses it (using 50th percentile instead). (3) The code loads some event times (sample, delay, reward, bitStart) that are not directly used for the output (bitStart is used only for video offset computation). (4) Full firing rate arrays are computed for ALL trials before filtering to valid trials.

ii.
```python
# Loads all trajectory data for both cameras
for cam in range(n_cams):
    ...
    for trix in range(cam_data['n_trials']):
        # loads ALL features for ALL trials

# Only uses tongue from cam 0 and top_paw from cam 1
tongue_vel = extract_feature_velocity(session, 0, 'tongue', time_axis, vidshift)
paw_vel = extract_feature_velocity(session, 1, 'top_paw', time_axis, vidshift)
```

iii. Loading all trajectory data wastes memory and I/O time, but does not affect correctness.
