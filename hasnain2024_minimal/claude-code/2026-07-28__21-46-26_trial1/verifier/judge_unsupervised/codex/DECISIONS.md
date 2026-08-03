# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script hard-codes two source directories, finds every `data_structure_*.mat` file in them, pairs each one with a same-name `motionEnergy_*.mat` file when present, and then processes each session one by one. Session `.mat` files are loaded with `h5py` first and with `scipy.io.loadmat` as a fallback for older MATLAB formats. Sessions with no `clu` field are treated as behavior-only and later skipped.

ii. Code snippets:
```python
DATA_DIRS = [
    '/app/data/Ephys_Behavior',
    '/app/data/RandomizedDelay_Ephys_Behavior',
]

def find_session_files(data_dirs):
    data_files = sorted(glob.glob(os.path.join(data_dir, 'data_structure_*.mat')))
    ...
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

iii. `CONVERSION_NOTES.md` says the agent intentionally used `Ephys_Behavior/` and `RandomizedDelay_Ephys_Behavior/`, loaded MATLAB v7.3 with `h5py` and v5/v7 with `scipy.io`, and skipped two behavior-only sessions. In the trajectory, the agent justified including both directories after reading the paper’s standard and randomized-delay task descriptions.

## 1-b. How are the data split into subjects?

i. Subjects are identified per session from the animal prefix in each filename, then deduplicated into a sorted `subjects` list. `subject_idx` maps each kept session to the index of its animal in that list.

ii. Code snippets:
```python
basename = os.path.basename(session_file)
parts = basename.replace('data_structure_', '').replace('.mat', '').split('_')
animal = parts[0]
```

```python
unique_subjects = sorted(set(all_animals))
subject_idx = np.array([unique_subjects.index(a) for a in all_animals], dtype=np.int64)
```

iii. The agent’s notes describe subject counts by dataset and mouse. The trajectory summary says this mirrors the paper/code convention of using `meta.anm` as the subject identifier.

## 1-c. How are the data split into sessions?

i. Each `data_structure_*.mat` file is treated as one session. The converted dataset stores one top-level session entry per successfully processed file.

ii. Code snippets:
```python
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
    ...
    all_neural.append(result['neural'])
```

iii. The notes report session counts separately for `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`, which shows the agent decided that one file equals one session.

## 1-d. How are the data split into trials?

i. Within each session, the script reads `bp.Ntrials` and represents trials by raw trial index. It then keeps only `valid_trials` after trial filtering, and each kept trial becomes one element in the session’s `neural`, `input`, and `output` lists.

ii. Code snippets:
```python
Ntrials = session['Ntrials']
...
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
    ...
    output_trials.append(output_trial)
```

iii. The notes say the agent matched the reference filtering logic but expanded it to include error trials for outcome decoding.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if they are non-stimulated, non-early, and have a behavioral response (`hit` or `miss`). Ignore/no-response trials are excluded, and sessions with fewer than two valid trials are dropped.

ii. Code snippets:
```python
trial_mask = (
    (session['stim_enable'] == 0) &
    (session['early'] == 0) &
    ((session['hit'] == 1) | (session['miss'] == 1))
)
...
if n_valid < 2:
    return None
```

iii. `CONVERSION_NOTES.md` explicitly states `~stim.enable`, `~early`, and `(hit | miss)`. The trajectory shows the agent deliberately diverged from the default MATLAB hit-only conditions because the decoder task required an `Outcome` output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from per-unit spike times and trial assignments in `obj.clu` (`trialtm`, `trial`, `quality`), plus trial-level `goCue` times for alignment and probe location metadata for ALM filtering.

ii. Code snippets:
```python
unit['quality'] = h5_read_string(f, f[q_ref])
unit['trialtm'] = f[tm_ref][()].flatten()
unit['trial'] = f[trial_ref][()].flatten().astype(int)
...
session['goCue'] = ev['goCue'][()].flatten()
```

```python
if probe_idx < len(session['probe_locations']):
    loc = session['probe_locations'][probe_idx]
```

iii. The trajectory summary ties these fields to `alignSpikes.m`, `findClusters.m`, and `getSeq.m`.

## 2-b. How is the `neural` data processed?

i. For each kept unit and kept trial, spikes are aligned to go cue, histogram-binned on a 5 ms grid from `-2.5` to `+2.5` s, converted to firing rates by dividing by `dt`, and smoothed with a causal Gaussian-like kernel intended to match MATLAB `mySmooth.m`.

ii. Code snippets:
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
...
aligned_times = spike_times[spike_mask] - go_time
counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts / dt
fr_smooth = causal_gaussian_smooth(fr, smooth_samples)
```

```python
def causal_gaussian_smooth(x, kernel_width_samples):
    kern = np.exp(-0.5 * ((n - center) / sigma) ** 2)
    kern[:N // 2] = 0
    kern = kern / kern.sum()
    return np.convolve(x, kern, mode='same')
```

iii. The notes say this follows `getDefaultParams.m`, `alignSpikes.m`, `getSeq.m`, and `mySmooth.m`, with `alignEvent='goCue'`, `dt=1/200`, `tmin=-2.5`, `tmax=2.5`, and `smooth=15`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script excludes non-ALM probes, removes units labeled `garbage`, `gabrga`, `noisy`, or `real?`, keeps unlabeled units, then removes units whose mean firing rate is `<= 0.5 Hz`.

ii. Code snippets:
```python
excluded = {'garbage', 'gabrga', 'noisy', 'real?'}
...
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

iii. The notes cite `findClusters.m` and `removeLowFRClusters.m`, and the trajectory summary says the ALM-only choice came from the paper’s statement that recordings analyzed here were from ALM.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every spike time in `trialtm` is shifted by subtracting that trial’s `goCue`, so each trial’s neural array is centered on go cue onset.

ii. Code snippets:
```python
go_time = go_cue_times[trial_idx]
...
aligned_times = spike_times[spike_mask] - go_time
```

iii. The notes and trajectory both identify `goCue` as the alignment event, matching `getDefaultParams.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Neural data use 5 ms bins (`DT = 1/200` s) over a 5 s window, giving 1000 time bins per trial. There is no second-stage rebinning; the initial binning defines the final resolution.

ii. Code snippets:
```python
DT = 1 / 200  # 5 ms bins
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
N_TIMEBINS = len(TIME_AXIS)
```

iii. `CONVERSION_NOTES.md` explicitly documents 5 ms bins and 1000 time bins.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a raw numeric field directly. It is constructed from the chosen alignment event (`goCue`) plus the fixed analysis window parameters `TMIN`, `TMAX`, and `DT`.

ii. Code snippets:
```python
ALIGN_EVENT = 'goCue'
TMIN = -2.5
TMAX = 2.5
DT = 1 / 200
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. The notes describe this as a continuous time axis relative to go cue, reflecting the decoder task rather than a separately stored raw signal.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script forms evenly spaced bin centers from `-2.5` to `+2.5` s and reuses the same 1-by-1000 vector for every trial.

ii. Code snippets:
```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
...
input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
```

iii. The justification in the notes is simply that the decoder input should be “Time from go cue onset (seconds)” in 5 ms steps.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is exactly co-registered with the neural data because neural binning, kinematic interpolation, motion-energy interpolation, and input construction all use the same `TIME_AXIS`.

ii. Code snippets:
```python
TIME_AXIS = EDGES[:-1] + DT / 2
...
trialdat = np.zeros((n_timebins, n_units, n_trials), dtype=np.float32)
...
input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
```

iii. The agent’s notes say the time input is the continuous axis used for the aligned neural window.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the raw behavioral variable `obj.bp.R`. Right trials are encoded as `1`; left trials are implicitly the complement and encoded as `0`.

ii. Code snippets:
```python
lick_direction = session['R'][valid_trials].astype(np.int32)  # R=1, L=0
```

iii. The notes state “R=1 (right), L=0 (left) from `obj.bp.R` field.”

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. After trial filtering, the script slices `R` on `valid_trials`, casts to `int32`, and tiles that per-trial constant across all time bins in the output matrix.

ii. Code snippets:
```python
lick_direction = session['R'][valid_trials].astype(np.int32)
...
output_trial[0, :] = lick_direction[ti]
```

iii. The agent justified this as the direct mapping required by the decoder spec.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`, where the agent interprets `autowater=1` as WC and `autowater=0` as DR.

ii. Code snippets:
```python
context = (1 - session['autowater'][valid_trials]).astype(np.int32)  # DR=1, WC=0
```

iii. The notes explicitly say context comes from `autowater`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The script converts the raw autowater flag into the target coding by computing `1 - autowater`, then tiles the result across time bins per trial.

ii. Code snippets:
```python
context = (1 - session['autowater'][valid_trials]).astype(np.int32)
...
output_trial[1, :] = context[ti]
```

iii. The notes justify this as “DR=1, WC=0 from `obj.bp.autowater`.”

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`; misses are included by the trial filter but encoded indirectly because `hit=0` on miss trials.

ii. Code snippets:
```python
trial_mask = ... ((session['hit'] == 1) | (session['miss'] == 1))
...
outcome = session['hit'][valid_trials].astype(np.int32)  # correct=1, incorrect=0
```

iii. The notes say the agent included both `hit` and `miss` trials specifically so outcome could be decoded.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The script slices `hit` on valid trials, casts to `int32`, and tiles the per-trial value across time bins.

ii. Code snippets:
```python
outcome = session['hit'][valid_trials].astype(np.int32)
...
output_trial[2, :] = outcome[ti]
```

iii. The agent’s justification is the decoder spec: `correct = 1`, `incorrect = 0`.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from camera-0 trajectory data: the `tongue` feature’s `x` and `y` coordinates in `obj.traj`, together with `frameTimes`, `NdroppedFrames`, the session video shift, and `goCue` for alignment.

ii. Code snippets:
```python
tongue_vel = extract_kinematic_feature(session, 0, 'tongue', valid_trials, go_cue_times)
...
x_pos = ts[feat_idx, 0, :].copy()
y_pos = ts[feat_idx, 1, :].copy()
frame_times = trial_data.get('frameTimes')
```

iii. The notes say the agent chose the side camera (`cam 0`) and the exact `tongue` feature after inspecting the trajectory structure and the MATLAB kinematics helpers.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The script locates the `tongue` trajectory, takes framewise gradients in `x` and `y` at 400 Hz, computes velocity magnitude, forces tongue-not-visible frames to zero, aligns frame times to go cue with a video offset correction, interpolates onto the neural time axis, and zero-fills entirely missing trials.

ii. Code snippets:
```python
dx = np.gradient(x_pos, dt_vid)
dy = np.gradient(y_pos, dt_vid)
vel = np.sqrt(dx**2 + dy**2)
...
nan_mask = np.isnan(x_pos) | np.isnan(y_pos)
vel[nan_mask] = 0.0
vel[np.isnan(vel)] = 0.0
```

```python
aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
interp_fn = interp1d(aligned_frame_times, vel, kind='linear',
                     bounds_error=False, fill_value=np.nan)
velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. The notes say this was intended to follow `findPosition.m` and the methods statement that missing values are not filled for tongue. The agent then chose to encode non-visible tongue frames as zero velocity to avoid NaNs in the decoder dataset.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The continuous tongue-velocity matrix is thresholded within each session at the 50th percentile over all values. Samples below threshold become `0`; samples at or above threshold become `1`.

ii. Code snippets:
```python
def discretize_per_session(values, percentile=50):
    threshold = np.percentile(valid, percentile)
    discretized = (values >= threshold).astype(np.int32)
```

```python
tongue_vel_disc = discretize_per_session(tongue_vel)
```

iii. The notes say this was done “Per decoder task specification.”

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The script uses each trial’s video `frameTimes`, subtracts a session-wide video-neural offset (`vidshift`) and that trial’s `goCue`, and interpolates the tongue-velocity trace onto the same `TIME_AXIS` used for neural data.

ii. Code snippets:
```python
aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
...
velocities[:, ti] = interp_fn(TIME_AXIS)
```

```python
bitstart_sglx = np.nanmedian(sglx['bitcode']['bitstart'][()].flatten())
bitStart_bp = np.nanmedian(ev['bitStart'][()].flatten())
session['vidshift'] = bitstart_sglx / fs - bitStart_bp
```

iii. The notes justify this by citing `findVideoOffset.m` and `findPosition.m`. The trajectory shows the agent believed it was matching `frameTimes - vidshift - obj.bp.ev.(alignEv)` from the MATLAB code.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera (`cam 1`) trajectory features `top_paw` and `bottom_paw`, using their `x/y` positions, `frameTimes`, `NdroppedFrames`, `goCue`, and `vidshift`.

ii. Code snippets:
```python
paw_top = extract_kinematic_feature(session, 1, 'top_paw', valid_trials, go_cue_times)
paw_bot = extract_kinematic_feature(session, 1, 'bottom_paw', valid_trials, go_cue_times)
```

iii. The notes say this choice came from the methods statement that paws were tracked using only the bottom view.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature, missing positions are filled with nearest valid values, velocity magnitude is computed from first derivatives at 400 Hz, the trace is aligned/interpolated to the neural axis, and when both `top_paw` and `bottom_paw` are available the two velocities are averaged.

ii. Code snippets:
```python
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

iii. The notes say nearest-neighbor filling was chosen to match the methods/reference for non-tongue features. Averaging the two paw tracks was the agent’s own simplification.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is discretized per session using the 50th percentile over all paw-velocity values, with `< threshold` mapped to `0` and `>= threshold` mapped to `1`.

ii. Code snippets:
```python
paw_vel_disc = discretize_per_session(paw_vel)
```

iii. The notes say this is per the decoder task spec.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity uses the same alignment pipeline as tongue velocity: `frameTimes - vidshift - goCue`, then interpolation onto `TIME_AXIS`.

ii. Code snippets:
```python
aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. The notes cite `findPosition.m` and `findVideoOffset.m` as the intended reference behavior.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_*.mat` file’s `me.data`, plus video `frameTimes` from `obj.traj`, trial `goCue` times, and the computed video-neural shift.

ii. Code snippets:
```python
me = load_motion_energy(me_file) if me_file else None
...
me_trial = me_data[trial_idx]
...
ft = session['traj'][0]['trials'][trial_idx].get('frameTimes')
```

iii. The notes explicitly say motion energy comes from the paired `motionEnergy_*.mat` files and is aligned with frame times and video offset.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The script loads each trial’s motion-energy trace, uses trajectory frame times when available, otherwise synthesizes 400 Hz frame times, aligns by `vidshift` and `goCue`, interpolates onto the neural time axis, and fills remaining NaNs with nearest values or zeros if an entire trace is missing.

ii. Code snippets:
```python
if frame_times is None:
    frame_times = np.arange(1, n_frames + 1) / 400.0
    aligned_frame_times = frame_times - 0.5 - go_time
else:
    aligned_frame_times = frame_times - vidshift - go_time
```

```python
interp_fn = interp1d(aligned_frame_times, me_trial,
                     kind='linear', bounds_error=False, fill_value=np.nan)
me_interp[:, ti] = interp_fn(TIME_AXIS)
...
col[mask] = np.interp(np.where(mask)[0], valid_idx, col[valid_idx])
```

iii. The notes say this was based on `loadMotionEnergy.m`, including nearest-value fill after interpolation.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized per session at the 50th percentile of all aligned motion-energy values.

ii. Code snippets:
```python
me_disc = discretize_per_session(me_interp)
```

iii. The notes say this follows the decoder task, not the paper’s manual motion-energy threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion-energy traces are aligned by subtracting `vidshift` and each trial’s `goCue`, then interpolating to the shared neural `TIME_AXIS`.

ii. Code snippets:
```python
aligned_frame_times = frame_times - vidshift - go_time
...
me_interp[:, ti] = interp_fn(TIME_AXIS)
```

iii. The notes justify this with the MATLAB `loadMotionEnergy.m` logic and the `findVideoOffset.m` offset formula.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script is broadly fault-tolerant. It uses many `try/except` blocks, defaults missing `stim`/`no` fields to zeros, skips probes with unreadable structures, sets `vidshift=0` if it cannot be computed, creates synthetic frame times if needed, nearest-fills many missing kinematic/motion-energy values, converts fully missing kinematic or motion-energy outputs to all zeros, and skips whole sessions only when they are behavior-only, have fewer than two valid trials, or have no remaining units.

ii. Code snippets:
```python
except:
    session['stim_enable'] = np.zeros(Ntrials)
...
except:
    session['vidshift'] = 0.0
```

```python
elif mask.all():
    velocities[:, ti] = 0.0
...
elif mask.all():
    me_interp[:, ti] = 0.0
```

```python
if me_interp is not None:
    me_disc = discretize_per_session(me_interp)
else:
    me_disc = np.zeros((N_TIMEBINS, n_trials), dtype=np.int32)
```

iii. `CONVERSION_NOTES.md` documents several concrete cases: behavior-only sessions are skipped, missing motion-energy files become all-zero outputs, some malformed paw sessions end up degenerate, and some late trials have all-zero neural data rather than being removed.

## 11-a. What are the most time-consuming steps of the code?

i. The dominant costs are the nested unit-by-trial spike binning/smoothing loop, the per-trial interpolation loops for kinematic and motion-energy data, and the session-by-session MATLAB loading/parsing.

ii. Code snippets:
```python
for ui, ci in enumerate(cluster_indices):
    ...
    for ti, trial_idx in enumerate(valid_trials):
        counts, _ = np.histogram(aligned_times, bins=edges)
        fr_smooth = causal_gaussian_smooth(fr, smooth_samples)
```

```python
for ti, trial_idx in enumerate(valid_trials):
    ...
    interp_fn = interp1d(aligned_frame_times, vel, ...)
```

iii. This is not stated in the notes, but it follows directly from the code structure.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the nested loops over units and trials in `align_and_bin_spikes`, the per-column NaN-filling loops in `extract_kinematic_feature` and `interpolate_motion_energy`, and the per-trial assembly loop that repeatedly copies the same `TIME_AXIS` and trialwise constants.

ii. Code snippets:
```python
for ui, ci in enumerate(cluster_indices):
    for ti, trial_idx in enumerate(valid_trials):
        ...
```

```python
for ti in range(n_trials):
    col = velocities[:, ti]
    ...
for ti in range(n_trials):
    col = me_interp[:, ti]
```

iii. This is an evaluator inference from the implementation; the agent did not explicitly discuss vectorization in its notes.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats the same align-and-interpolate pattern for tongue, paw, and motion energy; repeats exact feature-name scans for each extraction call; repeats NaN-filling column by column; and recreates the same `TIME_AXIS` input vector for every trial.

ii. Code snippets:
```python
tongue_vel = extract_kinematic_feature(...)
paw_top = extract_kinematic_feature(...)
paw_bot = extract_kinematic_feature(...)
me_interp = interpolate_motion_energy(...)
```

```python
for i, fn in enumerate(feat_names):
    if fn.lower() == feat_name.lower():
        feat_idx = i
```

```python
input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
```

iii. This is again an inference from the code layout rather than an explicit justification from the notes.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads and stores several fields that are not used in downstream decoder construction (`L`, `sample`, `delay`, `no`, `moveThresh`), imports `gaussian_filter1d` and defines `compute_velocity` without using them, builds `all_cluster_indices` without using it, and normalizes non-ALM labels to `tjM1` even though non-ALM probes are excluded.

ii. Code snippets:
```python
from scipy.ndimage import gaussian_filter1d
...
def compute_velocity(positions, dt_video=1/400):
```

```python
session['L'] = ...
session['sample'] = ...
session['delay'] = ...
session['no'] = ...
...
all_cluster_indices = []
```

iii. These are implementation leftovers rather than decisions justified in the notes. The notes do show that `moveThresh` was loaded but the final output uses a new per-session median split instead.
