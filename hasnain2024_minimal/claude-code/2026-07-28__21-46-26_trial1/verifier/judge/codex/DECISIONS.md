# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads session files by scanning two directories, pairing each `data_structure_*.mat` file with an optional `motionEnergy_*.mat` file, then processing each session file independently. It tries HDF5 (`h5py`) first and falls back to MATLAB v5/v7 loading with `scipy.io.loadmat`. Sessions with no neural `clu` field are skipped.

ii. ```python
DATA_DIRS = [
    '/app/data/Ephys_Behavior',
    '/app/data/RandomizedDelay_Ephys_Behavior',
]

data_files = sorted(glob.glob(os.path.join(data_dir, 'data_structure_*.mat')))
me_file = os.path.join(data_dir, f'motionEnergy_{parts}.mat')
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
    except (OSError, ValueError):
        pass
    return load_session_data_v5(filepath)
```

iii. `CONVERSION_NOTES.md` says the pipeline uses HDF5 loading with `h5py`, falls back to `scipy.io`, loads motion energy from separate files, and skips two behavior-only sessions. The trajectory summary repeats that the data mix MATLAB v7.3 and v5/v7 formats.

## 1-b. How are the data split into subjects?

i. The agent treats each unique animal identifier as a subject. It parses the animal ID from the session filename, collects all session-level animals, then builds `subjects` as a sorted unique list and `subject_idx` as an index per retained session.

ii. ```python
basename = os.path.basename(session_file)
parts = basename.replace('data_structure_', '').replace('.mat', '').split('_')
animal = parts[0]
```

```python
unique_subjects = sorted(set(all_animals))
subject_idx = np.array([unique_subjects.index(a) for a in all_animals], dtype=np.int64)
```

iii. The trajectory summary states that `/app/converted_data.pkl` contains session-level data and that subject assignment is derived from per-session animal identity. The code relies on filename parsing more than loaded metadata.

## 1-c. How are the data split into sessions?

i. Each `data_structure_*.mat` file is treated as one session. `convert_data()` iterates file pairs, calls `process_session()` once per file, and appends one session entry each to `neural`, `input`, and `output`.

ii. ```python
for df in data_files:
    sessions.append({
        'data_file': df,
        'me_file': me_file,
        'animal_date': parts,
    })
```

```python
for idx, sf in enumerate(session_files):
    result = process_session(sf['data_file'], sf['me_file'], idx)
    if result is None:
        continue
    all_neural.append(result['neural'])
```

iii. `CONVERSION_NOTES.md` describes counts in terms of sessions loaded from the two directories, and the trajectory summary says the final dataset contains 45 processed sessions.

## 1-d. How are the data split into trials?

i. Within each session, the agent builds a boolean `trial_mask`, converts it to `valid_trials`, and only retained trial indices become trial entries. Neural data are stored per retained trial, and each retained trial gets one input array and one output array.

ii. ```python
trial_mask = (
    (session['stim_enable'] == 0) &
    (session['early'] == 0) &
    ((session['hit'] == 1) | (session['miss'] == 1))
)
valid_trials = np.where(trial_mask)[0]
```

```python
for ti in range(n_trials):
    neural_trial = neural_data[:, :, ti].T.astype(np.float32)
    input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
    output_trial = np.zeros((6, N_TIMEBINS), dtype=np.int32)
```

iii. The notes say trials are filtered to non-stim, non-early, responding trials, and the trajectory summary describes the resulting dataset in terms of retained trial counts.

## 1-e. How are trials filtered based on quality controls?

i. The agent excludes optogenetically stimulated trials, early-lick trials, and non-responding trials. It retains both hit and miss trials, so incorrect trials remain available for outcome decoding. Sessions with fewer than 2 retained trials are dropped.

ii. ```python
trial_mask = (
    (session['stim_enable'] == 0) &
    (session['early'] == 0) &
    ((session['hit'] == 1) | (session['miss'] == 1))
)
...
if n_valid < 2:
    return None
```

iii. `CONVERSION_NOTES.md` explicitly calls these “matching reference code conditions” and lists `~stim.enable`, `~early`, and `(hit | miss)`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data come from per-unit spike times and per-spike trial assignments in `obj.clu`, plus go-cue times from `obj.bp.ev.goCue`. Probe location metadata are also used to keep only ALM probes.

ii. ```python
unit['trialtm'] = f[tm_ref][()].flatten()
unit['trial'] = f[trial_ref][()].flatten().astype(int)
session['goCue'] = ev['goCue'][()].flatten()
```

```python
if probe_idx < len(session['probe_locations']):
    loc = session['probe_locations'][probe_idx]
if 'ALM' not in loc_upper:
    continue
```

iii. The trajectory summary says the agent followed `alignSpikes.m`, `findClusters.m`, and `removeLowFRClusters.m`, all of which operate on cluster spike times, trial IDs, event times, and probe identity.

## 2-b. How is the `neural` data processed?

i. For each kept unit and retained trial, the agent subtracts go-cue time from spike times, bins aligned spikes from `-2.5` to `+2.5` s in 5 ms bins, converts counts to firing rates by dividing by `dt`, then applies a causal Gaussian smoother with kernel length 15 samples.

ii. ```python
TMIN = -2.5
TMAX = 2.5
DT = 1 / 200
SMOOTH_MS = 15
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
```

```python
aligned_times = spike_times[spike_mask] - go_time
counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts / dt
fr_smooth = causal_gaussian_smooth(fr, smooth_samples)
```

iii. The notes cite `getDefaultParams.m`, `getSeq.m`, and `mySmooth.m`; the trajectory summary says the agent intentionally matched `alignEvent='goCue'`, `dt=1/200`, `tmin=-2.5`, `tmax=2.5`, and `smooth=15`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent applies three QC filters to neural data: probe location must contain ALM, unit quality must not be in an excluded set (`garbage`, `gabrga`, `noisy`, `real?`), and mean firing rate must exceed 0.5 Hz.

ii. ```python
excluded = {'garbage', 'gabrga', 'noisy', 'real?'}
if q not in excluded and q != '':
    indices.append(i)
elif q == '':
    indices.append(i)
```

```python
if 'ALM' not in loc_upper:
    continue
...
fr_mask = mean_frs > LOW_FR
trialdat_probe = trialdat_probe[:, fr_mask, :]
```

iii. The notes say this matches `findClusters.m` with `quality='all'`, `removeLowFRClusters.m` with `lowFR=0.5`, and the paper’s focus on ALM recordings.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to go-cue onset by subtracting `goCue[trial_idx]` from each spike time in that trial before binning. The resulting bins are centered on a shared `TIME_AXIS` spanning `[-2.4975, 2.4975]` s relative to go cue.

ii. ```python
go_time = go_cue_times[trial_idx]
aligned_times = spike_times[spike_mask] - go_time
counts, _ = np.histogram(aligned_times, bins=edges)
```

```python
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. Both the notes and trajectory summary say the selected alignment event is `goCue`, matching the instruction “Temporally align based on Go cue onset.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 5 ms bins (`DT = 1/200` s). The agent does not perform any later rebinning; it bins spikes directly onto the final 5 ms grid.

ii. ```python
DT = 1 / 200  # 5 ms bins
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. `CONVERSION_NOTES.md` states “5 ms bins, giving 1000 time bins per trial,” and the trajectory summary cites `getDefaultParams.m` with `dt=1/200`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read as a raw signal; it is synthesized from the chosen alignment event and binning parameters. Conceptually it derives from go-cue alignment plus the constants `TMIN`, `TMAX`, and `DT`.

ii. ```python
TMIN = -2.5
TMAX = 2.5
DT = 1 / 200
TIME_AXIS = EDGES[:-1] + DT / 2
```

```python
input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
```

iii. The notes describe the decoder input as “Time from go cue onset (seconds): continuous time axis [-2.4975, ..., 2.4975] in 5ms steps.”

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The agent computes bin centers once from the shared edges array, reshapes that 1D vector to `(1, n_timebins)`, and reuses the same value for every trial in every session.

ii. ```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
...
input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
```

iii. This is justified in the notes as the continuous decoder input requested by the instructions, not as a raw measurement stream from the experiment.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is the same axis used to define the neural bin centers and to resample the kinematic and motion-energy streams. That makes it exactly co-registered with the neural data.

ii. ```python
TIME_AXIS = EDGES[:-1] + DT / 2
...
neural_trial = neural_data[:, :, ti].T.astype(np.float32)
input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
```

iii. The trajectory summary explicitly says the code uses one shared 5 ms axis for neural, kinematic, and motion-energy data after go-cue alignment.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the behavioral trial variable `R`, with the implicit convention that right is `1` and left is `0`. The code does not use `L` except to load it.

ii. ```python
session['R'] = bp['R'][()].flatten().astype(float)
session['L'] = bp['L'][()].flatten().astype(float)
...
lick_direction = session['R'][valid_trials].astype(np.int32)
```

iii. The notes state “Lick direction: R=1 (right), L=0 (left) from `obj.bp.R` field.”

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The agent subsets `R` to retained trials, casts to `int32`, and then tiles that per-trial category across all time bins in the output matrix.

ii. ```python
lick_direction = session['R'][valid_trials].astype(np.int32)
...
output_trial[0, :] = lick_direction[ti]
```

iii. The notes justify this as a per-trial categorical target consistent with the decoder task specification.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `autowater`. The code interprets `autowater=1` as water-cued (WC) and converts that to decoder label `0`; delayed-response (DR) becomes `1`.

ii. ```python
session['autowater'] = bp['autowater'][()].flatten().astype(float)
...
context = (1 - session['autowater'][valid_trials]).astype(np.int32)
```

iii. `CONVERSION_NOTES.md` explicitly says “Behavioral context: DR=1, WC=0 from `obj.bp.autowater` (autowater=1 means WC).”

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code inverts `autowater`, casts the result to integer labels, and repeats the resulting per-trial context value across time bins.

ii. ```python
context = (1 - session['autowater'][valid_trials]).astype(np.int32)
...
output_trial[1, :] = context[ti]
```

iii. The justification in the notes is that the decoder task requires WC=`0` and DR=`1`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `hit`, after trial filtering has already removed ignore/no-response trials and retained both hits and misses. That makes `hit=1` mean correct and `hit=0` correspond to retained miss trials.

ii. ```python
session['hit'] = bp['hit'][()].flatten().astype(float)
session['miss'] = bp['miss'][()].flatten().astype(float)
...
outcome = session['hit'][valid_trials].astype(np.int32)
```

iii. The notes say “Outcome: correct=1, incorrect=0 from `obj.bp.hit`,” and explain that miss trials are retained for decoding.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code subsets `hit` to retained trials, casts it to `int32`, and broadcasts that per-trial label over all time bins in the output tensor.

ii. ```python
outcome = session['hit'][valid_trials].astype(np.int32)
...
output_trial[2, :] = outcome[ti]
```

iii. The notes justify this as the required categorical correct/incorrect decoder output.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DeepLabCut trajectory data in `obj.traj`, specifically camera 0, feature name `tongue`, x/y positions, frame times, and go-cue times. Video-neural shift (`vidshift`) is also used for alignment.

ii. ```python
tongue_vel = extract_kinematic_feature(session, 0, 'tongue', valid_trials, go_cue_times)
...
x_pos = ts[feat_idx, 0, :].copy()
y_pos = ts[feat_idx, 1, :].copy()
```

iii. The notes say tongue velocity is extracted from the side camera, feature `tongue`, using DeepLabCut positions and interpolated to the neural axis.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code finds the exact `tongue` feature, extracts x/y positions, computes velocity magnitude with `np.gradient` at 400 Hz, sets velocity to `0` when the tongue is not visible, interpolates onto the neural time axis, and then fills any remaining all-NaN trial with zeros.

ii. ```python
dx = np.gradient(x_pos, dt_vid)
dy = np.gradient(y_pos, dt_vid)
vel = np.sqrt(dx**2 + dy**2)

if is_tongue:
    nan_mask = np.isnan(x_pos) | np.isnan(y_pos)
    vel[nan_mask] = 0.0
    vel[np.isnan(vel)] = 0.0
```

```python
aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
interp_fn = interp1d(aligned_frame_times, vel, kind='linear', bounds_error=False, fill_value=np.nan)
```

iii. The notes justify the special tongue handling by citing the paper’s statement that missing values were not nearest-filled for the tongue. The trajectory summary says the agent interpreted that as setting tongue velocity to zero when invisible.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The agent pools all tongue-velocity samples within a session, computes the 50th percentile, and labels values `< threshold` as `0` and `>= threshold` as `1`.

ii. ```python
def discretize_per_session(values, percentile=50):
    valid = values[~np.isnan(values)]
    threshold = np.percentile(valid, percentile)
    discretized = (values >= threshold).astype(np.int32)
```

```python
tongue_vel_disc = discretize_per_session(tongue_vel)
```

iii. `CONVERSION_NOTES.md` says tongue, paw, and motion energy were discretized into two bins using a per-session 50th percentile threshold, as required by the decoder task.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The agent aligns frame times by subtracting both video shift and go-cue time, then linearly interpolates tongue velocity onto the shared neural `TIME_AXIS`.

ii. ```python
aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
interp_fn = interp1d(aligned_frame_times, vel,
                    kind='linear', bounds_error=False, fill_value=np.nan)
velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. The notes and trajectory summary both say kinematic data were interpolated to the neural time axis after correcting for the video-neural offset from `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from DeepLabCut trajectory data in camera 1 (bottom view), specifically the `top_paw` and `bottom_paw` features, plus frame times, go-cue times, and video shift.

ii. ```python
paw_top = extract_kinematic_feature(session, 1, 'top_paw', valid_trials, go_cue_times)
paw_bot = extract_kinematic_feature(session, 1, 'bottom_paw', valid_trials, go_cue_times)
```

iii. The notes say paw velocity comes from the bottom camera and averages `top_paw` and `bottom_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The code extracts x/y paw positions, nearest-fills missing values for non-tongue features, computes velocity magnitude at 400 Hz, interpolates each paw track to the neural axis, and averages the `top_paw` and `bottom_paw` velocities when both are available.

ii. ```python
if not is_tongue:
    for pos in [x_pos, y_pos]:
        mask = np.isnan(pos)
        if mask.any() and not mask.all():
            pos[mask] = np.interp(np.where(mask)[0], valid_idx, pos[valid_idx])
```

```python
if paw_top is not None and paw_bot is not None:
    paw_vel = (paw_top + paw_bot) / 2.0
```

iii. The notes justify nearest-filling for non-tongue features by citing the methods text, and justify the bottom-camera choice from the same methods section.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The agent uses the same session-wise median split as for tongue velocity: pooled paw-velocity samples below the 50th percentile become `0`, and those at or above become `1`.

ii. ```python
paw_vel_disc = discretize_per_session(paw_vel)
```

iii. This is documented in the notes as a decoder-task-driven discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity is aligned exactly like tongue velocity: frame times are shifted by `vidshift` and `goCue`, then linearly interpolated to the neural `TIME_AXIS`.

ii. ```python
aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. The notes describe all kinematic features as interpolated to the neural time axis after video-offset correction.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_*.mat` file, specifically `me.data`, together with per-trial frame times from trajectory data when available and `vidshift`/`goCue` for alignment.

ii. ```python
mat = scipy.io.loadmat(filepath, squeeze_me=False)
me = mat['me']
data_field = me['data'][0, 0]
thresh_field = me['moveThresh'][0, 0]
```

```python
me_interp = interpolate_motion_energy(me['data'], session, valid_trials, go_cue_times)
```

iii. `CONVERSION_NOTES.md` says motion energy is loaded from separate `motionEnergy_*.mat` files and aligned using video frame times and video offset.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code loads precomputed per-trial motion-energy traces from `me.data`, interpolates them to the shared neural axis, nearest-fills missing interpolation gaps, and ignores the source file’s `moveThresh` for the final decoder variable.

ii. ```python
for i in range(data_field.shape[0]):
    for j in range(data_field.shape[1]):
        trial_me = data_field[i, j].flatten().astype(float)
        me_data.append(trial_me)
```

```python
interp_fn = interp1d(aligned_frame_times, me_trial,
                    kind='linear', bounds_error=False, fill_value=np.nan)
...
col[mask] = np.interp(np.where(mask)[0], valid_idx, col[valid_idx])
```

iii. The notes justify using interpolated motion-energy traces and then applying the decoder-task discretization, rather than reusing the paper’s manually chosen movement threshold.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The agent computes a per-session 50th percentile over all motion-energy samples and binarizes values by that threshold.

ii. ```python
me_disc = discretize_per_session(me_interp)
```

iii. The notes explicitly say motion energy is discretized into two bins using a per-session 50th percentile threshold because that is what the decoder task requested.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. When trajectory frame times are available, motion energy is aligned by subtracting `vidshift` and `goCue` from those frame times and interpolating onto `TIME_AXIS`. If frame times are missing, the code falls back to synthetic `1/400`-s frame times and a hard-coded `0.5`-s offset before subtracting `goCue`.

ii. ```python
if frame_times is None:
    frame_times = np.arange(1, n_frames + 1) / 400.0
    aligned_frame_times = frame_times - 0.5 - go_time
else:
    aligned_frame_times = frame_times - vidshift - go_time
```

iii. The notes only document the normal `vidshift`-based alignment path, while the code adds an undocumented fallback when frame times are missing.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent uses broad fallback handling: missing metadata become `'unknown'`, missing `stim`/`no` fields become zeros, probe-load failures produce empty probe lists, missing `vidshift` becomes `0.0`, missing paw/tongue/motion streams become all-zero outputs, all-NaN interpolated trials become zeros, and sessions with no neural `clu`, no valid units, or fewer than two valid trials are skipped.

ii. ```python
except:
    session['vidshift'] = 0.0
...
elif mask.all():
    velocities[:, ti] = 0.0
```

```python
if tongue_vel is not None:
    tongue_vel_disc = discretize_per_session(tongue_vel)
else:
    tongue_vel_disc = np.zeros((N_TIMEBINS, n_trials), dtype=np.int32)
```

iii. The notes call out known issues such as missing motion-energy files, behavior-only sessions, and all-zero neural trials. The trajectory summary also lists these as explicit fixes or documented caveats.

## 11-a. What are the most time-consuming steps of the code?

i. The most expensive work is the nested per-unit/per-trial spike alignment and binning, followed by per-trial interpolation for each kinematic stream and motion energy, plus repeated loading/parsing of MATLAB files.

ii. ```python
for ui, ci in enumerate(cluster_indices):
    ...
    for ti, trial_idx in enumerate(valid_trials):
        counts, _ = np.histogram(aligned_times, bins=edges)
```

```python
for ti, trial_idx in enumerate(valid_trials):
    ...
    interp_fn = interp1d(aligned_frame_times, vel, ...)
```

iii. This is inferred from the code structure; the trajectory summary also highlights spike binning, interpolation, and mixed MATLAB-format loading as the major implementation work.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are obvious vectorization candidates: unit-by-trial spike histogramming, convolution over units, per-trial interpolation/fill loops for tongue/paw/motion energy, and the repeated appending/flattening of motion-energy trials from object arrays.

ii. ```python
for ui, ci in enumerate(cluster_indices):
    for ti, trial_idx in enumerate(valid_trials):
        ...
```

```python
for ti in range(n_trials):
    col = velocities[:, ti]
    mask = np.isnan(col)
    ...
```

iii. The code does most operations in Python loops rather than batching by unit or by trial, and nothing in the notes suggests the agent intentionally optimized this path.

## 11-c. What processing does the code repeat multiple times?

i. The code repeatedly constructs the same `TIME_AXIS` input trial for every trial, performs nearly identical interpolation/fill logic separately for tongue, paw, and motion energy, and repeats nearest-fill passes column-by-column after interpolation.

ii. ```python
for ti in range(n_trials):
    input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
    input_trials.append(input_trial)
```

```python
col[mask] = np.interp(np.where(mask)[0], valid_idx, col[valid_idx])
```

iii. This is visible directly in `process_session()`, `extract_kinematic_feature()`, and `interpolate_motion_energy()`. The trajectory summary does not claim any deduplication or caching.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads several fields that never affect the final dataset (`L`, `sample`, `delay`, `no`, `moveThresh`, some metadata, `NdroppedFrames` except as a NaN gate), imports `gaussian_filter1d` and `sys` without using them, and normalizes non-ALM region labels even though non-ALM probes are already excluded.

ii. ```python
import sys
from scipy.ndimage import gaussian_filter1d
...
session['L'] = bp['L'][()].flatten().astype(float)
session['sample'] = ev['sample'][()].flatten() if 'sample' in ev else np.zeros(Ntrials)
session['delay'] = ev['delay'][()].flatten() if 'delay' in ev else np.zeros(Ntrials)
```

```python
threshold = float(thresh_field.flatten()[0])
return {'data': me_data, 'moveThresh': threshold}
```

iii. The notes focus on the retained decoder variables only; these extra fields are side effects of broad data loading rather than part of the downstream analysis output.
