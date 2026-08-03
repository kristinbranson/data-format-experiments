# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI scans loading scripts in `code/DataLoadingScripts/Recording and video/` to build a probe map (animal, date) -> probes. It then iterates over two data directories (`data/Ephys_Behavior/` and `data/RandomizedDelay_Ephys_Behavior/`), finding all files matching `data_structure_ANIMAL_DATE.mat`. It skips `MAH*` animals (behavior-only). Each .mat file is loaded using auto-format detection (HDF5 v7.3 or MATLAB v5) via `load_session_data()`. Motion energy is loaded separately from `motionEnergy_ANIMAL_DATE.mat` files.

ii.
```python
def get_available_sessions():
    sessions = []
    script_dir = 'code/DataLoadingScripts/Recording and video'
    probe_map = {}
    for f in sorted(os.listdir(script_dir)):
        if f.endswith('.m'):
            # parse probe info from loading scripts
            ...
    for data_dir in ['data/Ephys_Behavior', 'data/RandomizedDelay_Ephys_Behavior']:
        for fn in sorted(os.listdir(data_dir)):
            if not fn.startswith('data_structure_'):
                continue
            if animal.startswith('MAH'):
                continue
            ...
```

iii. The AI documented this in CONVERSION_NOTES.md Step 1-2: identified that the reference code loads sessions via loading scripts (e.g., `loadEKH1_ALMVideo.m`) that specify animal, date, and probe numbers. The AI replicated this by parsing the loading scripts to extract probe assignments, then directly loading each .mat file.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the animal name extracted from the filename pattern `data_structure_ANIMAL_DATE.mat`. After processing, unique subject names are collected and sorted. Each session is mapped to a subject via `subject_idx`.

ii.
```python
subjects = sorted(set(r['animal'] for r in results))
subject_idx = np.array([subjects.index(r['animal']) for r in results])
```

iii. The AI identified animals from filenames. This matches the reference approach where meta structures store `anm` (animal name) fields.

## 1-c. How are the data split into sessions?

i. Each .mat file corresponds to one session (one recording day for one animal). For animals with multiple probes, data from all probes in a session are concatenated into a single session entry (neurons combined). Sessions are processed individually via `process_session()`.

ii.
```python
# In process_session():
for p in probes:
    p_idx = p - 1
    fr, kept = process_spikes(session, p_idx, time_axis)
    if fr is not None and len(kept) > 0:
        all_firing_rates.append(fr)
firing_rates = np.concatenate(all_firing_rates, axis=0)  # combine probes
```

iii. The AI noted that the reference code treats each .mat file as one session. Multi-probe sessions (e.g., JEB15) have their probes concatenated into a single neuron pool per session.

## 1-d. How are the data split into trials?

i. Each session's behavioral data (`obj.bp`) contains `Ntrials` indicating total trials. Spike data is organized per-unit with trial indices. After filtering valid trials, each valid trial becomes one entry in the output lists, with neural data as `(n_neurons, n_timepoints)`.

ii.
```python
# In process_session():
for trial_idx in trial_indices:
    neural = firing_rates[:, :, trial_idx].astype(np.float32)
    neural_trials.append(neural)
```

iii. Trials are defined by the behavioral protocol data stored per session. The reference code similarly iterates over `obj.bp.Ntrials` trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials by excluding: (1) early lick trials (`bp['early']`), (2) stimulation trials (`bp['stim_enable']`), and (3) "no"/ignore trials (`bp['no']`). Session-level filtering requires >40 right hit DR trials AND >40 left hit DR trials.

ii.
```python
# Trial-level filtering:
valid_trials = ~bp['early'] & ~bp['stim_enable']
valid_trials = valid_trials & ~bp['no']

# Session-level filtering:
def check_session_inclusion(session_data, probes):
    r_hit_dr = bp['R'] & bp['hit'] & ~bp['stim_enable'] & ~bp['autowater'] & ~bp['early']
    l_hit_dr = bp['L'] & bp['hit'] & ~bp['stim_enable'] & ~bp['autowater'] & ~bp['early']
    return n_r > MIN_HIT_TRIALS and n_l > MIN_HIT_TRIALS, n_r, n_l
```

iii. The AI identified from `UseInclusionCritera.m` that sessions need >40 R and L hit DR trials. The reference code conditions filter `~stim.enable&~autowater&~early` for the inclusion check. For trial-level filtering, the AI chose to exclude early, stim, and no/ignore trials. The reference code's conditions (R&hit, L&hit, etc.) only use hit trials from specific conditions, but the reference code's `trialdat` in `getSeq.m` bins spikes for ALL `obj.bp.Ntrials` trials (not just filtered ones). The AI's additional exclusion of `no` trials goes beyond the reference code, which processes all trials in `trialdat`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `obj.clu` (cluster/spike data). Each unit has: `tm` (absolute spike times), `trial` (trial number per spike), `trialtm` (within-trial spike times), and `quality` (cluster quality label).

ii.
```python
unit['tm'] = np.array(f[clu_group['tm'][u, 0]]).flatten()
unit['trial'] = np.array(f[clu_group['trial'][u, 0]]).flatten().astype(int)
unit['trialtm'] = np.array(f[clu_group['trialtm'][u, 0]]).flatten()
unit['quality'] = read_h5_string(f, clu_group['quality'][u, 0])
```

iii. Documented in CONVERSION_NOTES.md Step 1: spike data comes from `obj.clu{probe}(unit).trialtm` aligned to goCue.

## 2-b. How is the `neural` data processed?

i. Processing pipeline: (1) Align spike times to goCue: `trialtm - goCue[trial]`, (2) Bin spikes using histogram with edges from time axis (5ms bins, -2.5 to 2.5s), (3) Convert to firing rate by dividing by dt, (4) Smooth with causal Gaussian kernel (window size 15 bins).

ii.
```python
def process_spikes(session, probe_idx, time_axis):
    for i, u_idx in enumerate(valid_units):
        for j in range(Ntrials):
            aligned_times = trialtm[spk_mask] - goCue[j]
            counts, _ = np.histogram(aligned_times, bins=edges)
            fr = counts.astype(np.float64) / DT
            fr_smooth = smooth_signal(fr, SMOOTH_N)
            firing_rates[i, :, j] = fr_smooth
```

iii. The AI identified from `alignSpikes.m` and `getSeq.m` that spikes are aligned to goCue, binned with `histc`, divided by dt to get firing rate, and smoothed with `mySmooth`. This matches the reference pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Remove 'garbage' quality clusters (matching `deleteGarbageClu.m`), (2) Remove units with mean firing rate <= 0.5 Hz (matching `removeLowFRClusters.m`). Sessions with <10 remaining units are excluded.

ii.
```python
# Stage 1: Remove garbage
for u_idx, unit in enumerate(units):
    quality = unit['quality'].lower().strip()
    if 'garbage' not in quality:
        valid_units.append(u_idx)

# Stage 2: Remove low FR
mean_frs = np.mean(np.mean(firing_rates, axis=2), axis=1)
fr_mask = mean_frs > LOW_FR

# Stage 3: Minimum units
if n_neurons < MIN_UNITS:
    return None
```

iii. The AI noted `deleteGarbageClu.m` removes garbage clusters, and `removeLowFRClusters.m` removes units with mean FR < `lowFR` (0.5 Hz). However, the reference code computes mean FR from `obj.psth` which is the PSTH averaged over condition-specific trials (R&hit, L&hit conditions), not all trials. The AI computes mean FR across ALL trials, which is a subtle difference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue: `aligned_times = trialtm - goCue[trial]`. The time axis spans -2.5 to 2.5 seconds from go cue onset.

ii.
```python
aligned_times = trialtm[spk_mask] - goCue[j]
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. Matches `alignSpikes.m`: `obj.clu{prbnum}(clu).trialtm_aligned = obj.clu{prbnum}(clu).trialtm - event` where event = `obj.bp.ev.goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Time bin size is 5ms (dt = 1/200 = 0.005s). Time axis spans -2.5 to 2.5s, giving 1001 time bins. No temporal rebinning is applied beyond the initial binning.

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

iii. Directly from `getDefaultParams.m`: `params.dt = 1/200`, `params.tmin = -2.5`, `params.tmax = 2.5`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is simply the time axis itself (relative to goCue), not derived from any specific raw data variable. It is the vector of time bin centers from -2.5 to 2.5 seconds.

ii.
```python
inp = time_axis.astype(np.float32).reshape(1, -1)  # (1, n_timepoints)
input_trials.append(inp)
```

iii. The AI constructs the time axis from parameters matching `getDefaultParams.m`. The time from go cue onset is the same for every trial since all trials are aligned to goCue and have the same time window.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing beyond constructing the time axis using `np.arange(TMIN, TMAX + DT/2, DT)`. The same time vector is used for all trials.

ii.
```python
time_axis = np.arange(TMIN, TMAX + DT/2, DT)
```

iii. This is a straightforward construction of equally-spaced time bins.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis is inherently aligned since both neural data and the input use the same time axis (bins centered on goCue). The neural spikes are binned using this same time axis as edges.

ii.
```python
edges = np.append(time_axis, time_axis[-1] + DT)
counts, _ = np.histogram(aligned_times, bins=edges)
# ...
inp = time_axis.astype(np.float32).reshape(1, -1)
```

iii. Both neural data and input share the same time axis by construction.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Derived from `obj.bp.R` (right lick indicator) and `obj.bp.L` (left lick indicator) - boolean arrays indicating lick direction per trial.

ii.
```python
bp['R'] = np.array(f['obj/bp/R']).flatten().astype(bool)
bp['L'] = np.array(f['obj/bp/L']).flatten().astype(bool)
# ...
lick_dir = 1 if bp['R'][trial_idx] else 0
```

iii. The AI identified `obj.bp.R` and `obj.bp.L` as directional indicators from the data structure exploration.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Binary encoding: right = 1, left = 0. This is a per-trial constant broadcast across all time points.

ii.
```python
lick_dir = 1 if bp['R'][trial_idx] else 0
out[0, :] = lick_dir  # constant across time
```

iii. Matches the instruction specification: "left = 0, right = 1, per-trial".

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `obj.bp.autowater` - a boolean array where True indicates water-cued (WC) trials and False indicates delayed-response (DR) trials.

ii.
```python
bp['autowater'] = np.array(f['obj/bp/autowater']).flatten().astype(bool)
# ...
context = 0 if bp['autowater'][trial_idx] else 1
```

iii. The AI identified `autowater` flag as the context indicator. This is consistent with the reference code conditions where `autowater` distinguishes WC from DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Binary encoding: WC (autowater=True) = 0, DR (autowater=False) = 1. Per-trial constant broadcast across all time points.

ii.
```python
context = 0 if bp['autowater'][trial_idx] else 1
out[1, :] = context
```

iii. Matches the instruction specification: "WC = 0, DR = 1, per-trial".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `obj.bp.hit` - a boolean array indicating correct (hit) trials.

ii.
```python
bp['hit'] = np.array(f['obj/bp/hit']).flatten().astype(bool)
# ...
outcome = 1 if bp['hit'][trial_idx] else 0
```

iii. The AI uses `hit` to determine correct trials, with all non-hit trials coded as incorrect.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Binary encoding: correct (hit) = 1, incorrect (not hit) = 0. Per-trial constant broadcast across all time points. Note: since `no` (ignore) trials are filtered out, the remaining non-hit trials are `miss` trials.

ii.
```python
outcome = 1 if bp['hit'][trial_idx] else 0
out[2, :] = outcome
```

iii. Matches the instruction specification: "incorrect = 0, correct = 1, per-trial".

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from `obj.traj{1}` (camera 0 / side camera) trajectory data. Specifically, the 'tongue' feature's x and y coordinates from the DLC tracking: `traj.ts[feat_idx, 0:2, :]` (x, y positions over time) and `traj.frameTimes` for temporal alignment.

ii.
```python
tongue_vel = extract_feature_velocity(session, 0, 'tongue', time_axis, vidshift)
# In extract_feature_velocity:
x = ts[feat_idx, 0, :]  # x coordinate
y = ts[feat_idx, 1, :]  # y coordinate
```

iii. The AI identified from `getKinematicsFromVideo.m` that tongue is from camera 0 (side camera), matching `params.traj_features{1}` which includes 'tongue'.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Processing: (1) Extract x, y positions from DLC tracking for 'tongue' feature, (2) Compute video offset using `findVideoOffset`, (3) Align frame times to goCue: `frameTimes - vidshift - goCue[trial]`, (4) Interpolate positions to neural time axis, (5) Compute velocity via `np.gradient()` on x and y, (6) Set NaN velocities to 0 (tongue not visible), (7) Compute velocity magnitude: `sqrt(xvel^2 + yvel^2)`.

ii.
```python
# Align and interpolate
aligned_times = frame_times - vidshift - goCue[trix]
f_x = interp1d(aligned_times[valid], x[valid], kind='linear', ...)
xpos[:, trix] = f_x(time_axis)
# Velocity
xv = np.gradient(xpos[:, trix])
# Tongue-specific: NaN -> 0
xv[np.isnan(xv)] = 0
yv[np.isnan(yv)] = 0
# Magnitude
vel_mag = np.sqrt(xvel**2 + yvel**2)
```

iii. Matches the reference code in `findPosition.m` (interpolation) and `findVelocity.m` (gradient, NaN->0 for tongue). The AI noted that tongue features are NOT smoothed before interpolation and NaN velocity is set to 0 (tongue not visible).

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile threshold computed from all valid trial data. Values >= threshold are coded as 1 (high), values < threshold as 0 (low). NaN values are set to 0.

ii.
```python
tongue_valid = tongue_vel[:, trial_indices]
tongue_thresh = np.nanpercentile(tongue_valid[~np.isnan(tongue_valid)], 50)
# ...
tv_disc = np.zeros(len(time_axis), dtype=np.int64)
tv_disc[tv >= tongue_thresh] = 1
tv_disc[np.isnan(tv)] = 0
```

iii. Follows instruction: "discretized into two bins with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile". However, the tongue velocity distribution is heavily skewed because NaN tongue values (tongue not visible) are set to 0 velocity, making the 50th percentile very small or zero. This causes extreme class imbalance (5.7% low, 94.3% high across the dataset).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are aligned to goCue with video offset correction, then positions are interpolated to the neural time axis using linear interpolation. Velocity is computed on the interpolated positions.

ii.
```python
aligned_times = frame_times - vidshift - goCue[trix]
f_x = interp1d(aligned_times[valid], x[valid], kind='linear', ...)
xpos[:, trix] = f_x(time_axis)
```

iii. Matches `findPosition.m`: `interp1(traj(trix).frameTimes-vidshift-obj.bp.ev.(alignEv)(trix), ts, taxis)`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from `obj.traj{2}` (camera 1 / top camera), feature 'top_paw'. Uses x, y coordinates and frame times.

ii.
```python
paw_vel = extract_feature_velocity(session, 1, 'top_paw', time_axis, vidshift)
```

iii. The AI identified from `getDefaultParams.m` that `params.traj_features{2}` includes 'top_paw' from camera 1.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as tongue but with non-tongue-specific handling: (1) Extract x, y from DLC, (2) Align and interpolate to neural time axis, (3) Fill NaN positions with nearest valid value, (4) Compute velocity via `np.gradient()`, (5) Subtract baseline derivative (median of diff), (6) Fill NaN velocity with nearest, (7) Compute magnitude.

ii.
```python
# Non-tongue: fill missing positions
if not is_tongue:
    xpos[:, trix] = _fill_nearest(xpos[:, trix])
    ypos[:, trix] = _fill_nearest(ypos[:, trix])
# Velocity with baseline subtraction
if not is_tongue:
    base_x = np.nanmedian(np.diff(xpos[:, trix]))
    base_y = np.nanmedian(np.diff(ypos[:, trix]))
    xv = xv - base_x
    yv = yv - base_y
    xv = _fill_nearest(xv)
    yv = _fill_nearest(yv)
```

iii. Matches `findVelocity.m`: baseline subtraction (`basederiv = median(diff(tsinterp),'omitnan')`) and `fillmissing(,'nearest')` for non-tongue features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: per-session 50th percentile. Values >= threshold -> 1, < threshold -> 0. NaN -> 0.

ii.
```python
paw_valid = paw_vel[:, trial_indices]
paw_thresh = np.nanpercentile(paw_valid[~np.isnan(paw_valid)], 50)
pv_disc = np.zeros(len(time_axis), dtype=np.int64)
pv_disc[pv >= paw_thresh] = 1
pv_disc[np.isnan(pv)] = 0
```

iii. Follows instructions. Paw velocity is more balanced (~50/50) since paw tracking is more consistently available.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times aligned to goCue with video offset, then interpolated to neural time axis.

ii.
```python
aligned_times = frame_times - vidshift - goCue[trix]
f_x = interp1d(aligned_times[valid], x[valid], ...)
xpos[:, trix] = f_x(time_axis)
```

iii. Same alignment approach as tongue velocity. Matches `findPosition.m`.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from separate `motionEnergy_ANIMAL_DATE.mat` files. The variable `me.data` contains per-trial motion energy time series (at 400 Hz video frame rate). `me.moveThresh` stores a movement threshold (not used for discretization here).

ii.
```python
def load_motion_energy(me_filepath):
    data = sio.loadmat(me_filepath, squeeze_me=False)
    me_raw = data['me']
    # Handle struct format: me.data, me.moveThresh
    # Handle double-nested: me.data.data
    ...
    return {'data': me_trials, 'moveThresh': me_thresh}
```

iii. From `loadMotionEnergy.m`: loads `motionEnergy_*.mat` files containing per-trial motion energy data.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Processing: (1) Load per-trial motion energy vectors, (2) Get frame times from camera 0 trajectory data, (3) Align to goCue using video offset: `frameTimes - vidshift - goCue[trial]`, (4) Interpolate to neural time axis, (5) Fill NaN with nearest valid value.

ii.
```python
def align_motion_energy(me_data, session, time_axis, vidshift):
    for trix in range(min(Ntrials, n_me_trials)):
        # Get frame times from camera 0
        frame_times = trial_traj['frameTimes']
        aligned_times = frame_times - vidshift - goCue[trix]
        f_me = interp1d(aligned_times[:n], trial_me[:n], kind='linear', ...)
        me_aligned[:, trix] = f_me(time_axis)
    # Fill NaN
    for trix in range(Ntrials):
        me_aligned[:, trix] = _fill_nearest(me_aligned[:, trix])
```

iii. Matches `loadMotionEnergy.m`: `interp1(obj.traj{1}(trix).frameTimes-vidshift-alignTimes(trix),me.data{trix},taxis)` followed by `fillmissing(me.data,'nearest')`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile threshold. Values >= threshold -> 1 (high), < threshold -> 0 (low). NaN -> 0.

ii.
```python
me_valid = me_aligned[:, trial_indices]
me_thresh = np.nanpercentile(me_valid[~np.isnan(me_valid)], 50)
me_disc = np.zeros(len(time_axis), dtype=np.int64)
me_disc[me >= me_thresh] = 1
me_disc[np.isnan(me)] = 0
```

iii. Follows instructions. Motion energy is well-balanced (~50/50).

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from camera 0 are used for alignment: `frameTimes - vidshift - goCue[trial]`, then interpolated to neural time axis.

ii.
```python
aligned_times = frame_times - vidshift - goCue[trix]
f_me = interp1d(aligned_times[:n], trial_me[:n], ...)
me_aligned[:, trix] = f_me(time_axis)
```

iii. Matches `loadMotionEnergy.m`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple data quality issues are handled:
- **Missing trajectory data**: Trials with `NdroppedFrames = NaN` are skipped (trajectory set to NaN).
- **Missing frame times**: Fallback to synthetic frame times `np.arange(1, n_frames+1) / 400`.
- **Missing motion energy files**: Session gets all-NaN motion energy.
- **NaN tongue velocity**: Set to 0 (tongue not visible).
- **NaN paw/non-tongue velocity**: Filled with nearest valid value.
- **NaN positions (non-tongue)**: Filled with nearest valid value.
- **Failed HDF5 references**: Empty unit lists created for invalid probes.
- **All-zero neural trials**: Left in data (30 trials in sessions 35 and 42 have all-zero neural data from late trials where recording ended).

ii.
```python
# Missing trajectory
if trial['ts'] is None:
    continue
if np.isnan(trial['NdroppedFrames']):
    continue

# Missing frame times fallback
if trial_traj is None or trial_traj['frameTimes'] is None:
    frame_times = np.arange(1, n_frames + 1) / VIDEO_FS

# Invalid probes
if not isinstance(clu_group, h5py.Group) or 'tm' not in clu_group:
    clu_list.append([])
```

iii. The AI documented edge cases in CONVERSION_NOTES.md Step 10. Missing data handling follows the reference code patterns (e.g., `fillmissing('nearest')` in MATLAB becomes `_fill_nearest()` in Python).

## 11-a. What are the most time-consuming steps of the code?

i. Based on the conversion output timing:
- **Data loading** (~2-5s per session): Loading large HDF5/.mat files.
- **Spike processing** (~1-5s per session): Iterating over all units and trials to bin and smooth spikes.
- **Trajectory extraction** (~1-3s per session per feature): Interpolating DLC tracking data for tongue and paw.
- Total full conversion: ~303s (5.1 min) for 47 sessions.

ii.
```python
# Timing is printed per step:
print(f'    Loaded in {t_load:.1f}s')
print(f'    Spike processing: {t_spk:.1f}s')
print(f'    Tongue velocity: {t_tongue:.1f}s')
print(f'    Paw velocity: {t_paw:.1f}s')
print(f'    Motion energy: {t_me:.1f}s')
```

iii. CONVERSION_NOTES.md Step 7 estimated ~8s per session, ~5 min for full conversion.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The main inefficiency is the double loop in `process_spikes()` iterating over units and trials individually. This could be vectorized by processing all spikes for a unit at once using vectorized histogram operations. The `_fill_nearest` function also has a loop that could be vectorized using `np.interp` or pandas `fillna`.

ii.
```python
# Inner loop over trials in process_spikes:
for i, u_idx in enumerate(valid_units):
    for j in range(Ntrials):
        spk_mask = trial_nums == trial_num
        counts, _ = np.histogram(aligned_times, bins=edges)
        fr_smooth = smooth_signal(fr, SMOOTH_N)
        firing_rates[i, :, j] = fr_smooth

# _fill_nearest loop:
for i in range(len(arr)):
    if mask[i]:
        dists = np.abs(valid - i)
        arr[i] = arr[valid[np.argmin(dists)]]
```

iii. These loops are the primary bottleneck but the total runtime (~5 min) is acceptable.

## 11-c. What processing does the code repeat multiple times?

i. The video offset (`compute_video_offset`) is computed once per session, which is correct. However, the `extract_feature_velocity` function is called separately for tongue and paw, and each call independently loops through all trials to do interpolation - the frame time alignment computation is repeated. The smoothing kernel is recomputed for every unit/trial in `smooth_signal` via `causal_gaussian_kernel`.

ii.
```python
# Smoothing kernel recomputed every call:
def smooth_signal(x, N):
    kern = causal_gaussian_kernel(N)  # recomputed each time
    return np.convolve(x, kern, mode='same')
```

iii. The kernel could be computed once and reused.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code processes trajectory data for ALL trials before filtering to valid trials. Firing rates are computed for ALL `Ntrials` before selecting only `valid_trials`. This means spike binning/smoothing is done for early, stim, and no trials that are later discarded. Additionally, per-trial outputs for per-trial variables (lick direction, context, outcome) are broadcast to `(n_timepoints,)` arrays, using more memory than necessary for constant values.

ii.
```python
# Compute firing rates for ALL trials, then filter:
firing_rates = np.zeros((len(valid_units), n_time, Ntrials), dtype=np.float32)
# ...later:
for trial_idx in trial_indices:  # only use valid trials
    neural = firing_rates[:, :, trial_idx]
```

iii. Processing all trials before filtering is simpler but wasteful. The reference code similarly processes all trials in `getSeq.m` (`obj.trialdat` has size `(time, units, Ntrials)` for all trials).
