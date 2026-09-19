# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script discovers every `data_structure_*.mat` file under both `/app/data/Ephys_Behavior` and `/app/data/RandomizedDelay_Ephys_Behavior`, pairs it with a same-session `motionEnergy_*.mat` when present, and loads each session with `load_session_data`, which tries HDF5 first and then falls back to `scipy.io` for v5/v7 files. Sessions that fail or have no usable electrophysiology are skipped.

ii.

```python
DATA_DIRS = [
    '/app/data/Ephys_Behavior',
    '/app/data/RandomizedDelay_Ephys_Behavior',
]

for data_dir in data_dirs:
    data_files = sorted(glob.glob(os.path.join(data_dir, 'data_structure_*.mat')))
...
def load_session_data(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            ...
        return _load_session_data_h5(filepath)
    except (OSError, ValueError):
        pass
    return load_session_data_v5(filepath)
```

iii. In the trajectory the agent first considered restricting the dataset, then explicitly added support for both task folders, added a v5/v7 loader fallback, and decided to skip behavior-only sessions that lacked `clu` data (steps 21, 72, 77, 93, 105).

## 1-b. How are the data split into subjects (mice)?

i. Each session's subject is parsed from the filename stem before the first underscore, and the output `subjects` list is the sorted unique set of those animal IDs. `subject_idx` points each processed session back into that list.

ii.

```python
basename = os.path.basename(session_file)
parts = basename.replace('data_structure_', '').replace('.mat', '').split('_')
animal = parts[0]
...
unique_subjects = sorted(set(all_animals))
subject_idx = np.array([unique_subjects.index(a) for a in all_animals], dtype=np.int64)
```

iii. The trajectory does not give a separate rationale for subject parsing; the agent consistently treated the filename stem as the session identity throughout loading and reporting.

## 1-c. How are the data split into sessions?

i. One session is one discovered `data_structure_*.mat` file. Every successfully processed file becomes one element of `neural`, `input`, and `output`, regardless of whether it belongs to the fixed-delay or randomized-delay folder.

ii.

```python
def find_session_files(data_dirs):
    sessions = []
    ...
    sessions.append({
        'data_file': df,
        'me_file': me_file,
        'animal_date': parts,
    })
...
for idx, sf in enumerate(session_files):
    result = process_session(sf['data_file'], sf['me_file'], idx)
```

iii. The agent explicitly reasoned about including Ephys and RandomizedDelay sessions and later summarized the resulting counts by folder, confirming that its session definition was file-based rather than reference-list-based (steps 21, 93, 105).

## 1-d. How are the data split into trials?

i. Trials are indexed by positions in the per-trial behavioral arrays loaded from `bp`, then filtered with `trial_mask`. Neural spikes remain attached to trials through each unit's 1-indexed `trial` array.

ii.

```python
Ntrials = session['Ntrials']
...
trial_mask = (
    (session['stim_enable'] == 0) &
    (session['early'] == 0) &
    ((session['hit'] == 1) | (session['miss'] == 1))
)
valid_trials = np.where(trial_mask)[0]
...
spike_trials = unit['trial']
trial_num = trial_idx + 1
spike_mask = spike_trials == trial_num
```

iii. The trajectory repeatedly discusses per-trial behavioral fields and spike `trial` assignments, so the AI was relying on the file's existing trial structure rather than reconstructing trial boundaries (steps 21, 77).

## 1-e. How are trials filtered based on quality controls?

i. The script keeps only non-stimulation, non-early, responding trials: `stim_enable == 0`, `early == 0`, and `(hit == 1 or miss == 1)`. It therefore drops ignore/no-response trials entirely and does not explicitly remove trials that continue after the recording ends.

ii.

```python
trial_mask = (
    (session['stim_enable'] == 0) &
    (session['early'] == 0) &
    ((session['hit'] == 1) | (session['miss'] == 1))
)
valid_trials = np.where(trial_mask)[0]
```

iii. In step 21 the agent said it would exclude early-lick and ignore trials while keeping correct and error trials. Later verification output still showed late all-zero trials, indicating that it never added the recording-length cut used by the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each cluster's per-spike `trialtm` and `trial` arrays, with `goCue` providing the event used for alignment. Cluster quality labels and probe locations are used for later filtering.

ii.

```python
unit['trialtm'] = f[tm_ref][()].flatten()
unit['trial'] = f[trial_ref][()].flatten().astype(int)
...
go_time = go_cue_times[trial_idx]
aligned_times = spike_times[spike_mask] - go_time
```

iii. The trajectory consistently describes the neural pipeline as extracting spike times from the cluster structures, aligning them to go cue, and then binning them (step 21).

## 2-b. How is the `neural` data processed?

i. For each unit and valid trial, the script aligns spike times to go cue, bins them into 5 ms bins, converts counts to Hz, and applies a causal Gaussian convolution with a 15-sample kernel via `causal_gaussian_smooth`.

ii.

```python
counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts / dt
fr_smooth = causal_gaussian_smooth(fr, smooth_samples)
trialdat[:, ui, ti] = fr_smooth
```

iii. The agent explicitly planned to align spikes to go cue, bin at 5 ms, and keep the existing 0.5 Hz / causal-smoothing style because it believed that matched the source code it had read (step 21).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script keeps only probes whose location string contains `ALM`, excludes clusters labeled `garbage`, `gabrga`, `noisy`, or `real?`, keeps empty quality labels, and then removes units whose mean firing rate across all trials and time bins is not above 0.5 Hz.

ii.

```python
excluded = {'garbage', 'gabrga', 'noisy', 'real?'}
...
if 'ALM' not in loc_upper:
    continue
...
fr_mask = mean_frs > LOW_FR
LOW_FR = 0.5
```

iii. Step 33 says the agent wanted ALM-only probes because the paper recorded from ALM, and step 21 says it chose 0.5 Hz because it thought the existing code used that threshold. Step 105 later acknowledged that the paper may actually use 1 Hz.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned by subtracting each trial's `goCue` time from the spike times already stored relative to trial start, then histogramming the aligned times into the shared bin edges.

ii.

```python
go_time = go_cue_times[trial_idx]
aligned_times = spike_times[spike_mask] - go_time
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The trajectory repeatedly describes go-cue alignment as the central temporal alignment choice for the dataset (step 21).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural grid uses 5 ms bins (`DT = 1/200`) over `[-2.5, 2.5]` s around go cue, for 1000 bins per trial. No further temporal rebinning is applied to the neural signal after that.

ii.

```python
TMIN = -2.5
TMAX = 2.5
DT = 1 / 200
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
N_TIMEBINS = len(TIME_AXIS)
```

iii. The trajectory and code both anchor the pipeline on 5 ms bins spanning a 5 s window centered on go cue (step 21).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This input is not read from a stored raw variable. It is constructed from the chosen alignment window and bin size around the go cue, producing the shared `TIME_AXIS` used for every trial.

ii.

```python
EDGES = np.arange(TMIN, TMAX + DT / 2, DT)
TIME_AXIS = EDGES[:-1] + DT / 2
```

iii. The trajectory treats this as the decoder's synthetic time base rather than a directly recorded signal.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script computes bin centers from the shared edges and reshapes that 1D axis into `(1, n_timebins)` for every trial.

ii.

```python
TIME_AXIS = EDGES[:-1] + DT / 2
...
input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
```

iii. No separate trajectory justification was given beyond defining the common neural time grid.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is exactly the center of the same bins used for neural histogramming, so input time bin `k` corresponds to the same interval as neural time bin `k`.

ii.

```python
counts, _ = np.histogram(aligned_times, bins=edges)
...
input_trial = TIME_AXIS.reshape(1, -1).astype(np.float32)
```

iii. The trajectory describes one shared go-cue-aligned time axis for neural and behavioral streams (step 21).

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The final code derives lick direction directly from the behavioral `R` flag on the kept trials, implicitly treating right-target trials as right licks and all others as left licks.

ii.

```python
lick_direction = session['R'][valid_trials].astype(np.int32)  # R=1, L=0
```

iii. The trajectory never gives a final explicit defense of this simplification. Earlier reasoning discussed needing lick direction from the behavioral table, but the final code collapses that to the `R` field after ignore trials have already been removed.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. No extra logic is applied beyond casting `R` to integers and repeating that per-trial label across all time bins.

ii.

```python
output_trial = np.zeros((6, N_TIMEBINS), dtype=np.int32)
output_trial[0, :] = lick_direction[ti]
```

iii. No later trajectory step revisits the final lick-direction rule; the code's behavior is simply the direct consequence of using `R` as the label.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `autowater` flag.

ii.

```python
context = (1 - session['autowater'][valid_trials]).astype(np.int32)
```

iii. The trajectory explicitly notes that `autowater=1` corresponds to WC context and uses that field to distinguish WC from DR (steps 16, 21).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. It relabels `autowater` so WC becomes 0 and DR becomes 1 by computing `1 - autowater`.

ii.

```python
context = (1 - session['autowater'][valid_trials]).astype(np.int32)  # DR=1, WC=0
```

iii. The trajectory explicitly mapped `autowater=1` to WC and the complement to DR (steps 16, 21).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived only from the `hit` flag on the already filtered, responding trials.

ii.

```python
outcome = session['hit'][valid_trials].astype(np.int32)  # correct=1, incorrect=0
```

iii. In step 21 the agent said it wanted correct and error trials and had already removed ignore trials. The final code therefore collapses outcome to hit versus non-hit among those kept trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The script casts `hit` to integers, so hits become 1 and all other kept trials become 0, then repeats that label across time.

ii.

```python
outcome = session['hit'][valid_trials].astype(np.int32)
...
output_trial[2, :] = outcome[ti]
```

iii. The trajectory justification is the same as 6-a: the agent wanted to retain correct and error trials while excluding ignores, which led to a binary outcome.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from one trajectory stream: camera 0, feature name `tongue`, using its `ts` positions, `frameTimes`, the session video shift, and each trial's `goCue`.

ii.

```python
tongue_vel = extract_kinematic_feature(session, 0, 'tongue', valid_trials, go_cue_times)
...
ts = trial_data.get('ts')
frame_times = trial_data.get('frameTimes')
vidshift = session.get('vidshift', 0.0)
```

iii. The trajectory shows the agent debugging the side-camera tongue feature specifically and deciding that invisible-tongue periods should become zero velocity (steps 37, 52, 55).

## 7-b. How is `output` *Tongue velocity* processed?

i. The code extracts x/y traces, computes velocity by `np.gradient` at 400 Hz, forces NaN tongue velocity to zero, linearly interpolates that frame-rate velocity onto the neural time axis, then fills any remaining NaNs with nearest values or zeros.

ii.

```python
x_pos = ts[feat_idx, 0, :].copy()
y_pos = ts[feat_idx, 1, :].copy()
dx = np.gradient(x_pos, dt_vid)
dy = np.gradient(y_pos, dt_vid)
vel = np.sqrt(dx**2 + dy**2)
...
vel[nan_mask] = 0.0
...
interp_fn = interp1d(aligned_frame_times, vel, kind='linear', bounds_error=False, fill_value=np.nan)
velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. Step 37 says the agent concluded that the tongue is usually NaN when not visible and should therefore be treated as zero velocity. Steps 52 and 55 show it accepting the resulting median-split degeneracy rather than changing the pipeline.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. It is split into only two categories by the session-wide 50th percentile, with values below threshold mapped to 0 and values at or above threshold mapped to 1. There is no separate `not visible` class.

ii.

```python
def discretize_per_session(values, percentile=50):
    threshold = np.percentile(valid, percentile)
    discretized = (values >= threshold).astype(np.int32)
    return discretized
```

iii. Steps 52, 53, and 55 show the agent reasoning that tongue velocity often has median 0 and then deciding to keep the prompt's 50th-percentile rule anyway, even though it made the output degenerate.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are shifted by `vidshift` and `goCue`, then the velocity trace is linearly interpolated onto `TIME_AXIS`. If frame times are missing, synthetic `1/400` s frame times are generated first.

ii.

```python
if frame_times is None or np.all(np.isnan(frame_times)):
    frame_times = np.arange(1, n_frames + 1) / 400.0
aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
interp_fn = interp1d(aligned_frame_times, vel, kind='linear', bounds_error=False, fill_value=np.nan)
velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. The trajectory states that video signals should be aligned to go cue with a video offset and interpolated to the neural time axis (step 21), and later accepts fallback handling for missing frame times.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera trajectory features `top_paw` and `bottom_paw`, using the same `ts`, `frameTimes`, `vidshift`, and `goCue` machinery as the tongue.

ii.

```python
paw_top = extract_kinematic_feature(session, 1, 'top_paw', valid_trials, go_cue_times)
paw_bot = extract_kinematic_feature(session, 1, 'bottom_paw', valid_trials, go_cue_times)
```

iii. The trajectory only briefly notes that the agent needed to handle paw features too; the final code shows that it chose to use both bottom-camera paw tracks.

## 8-b. How is `output` *Paw velocity* processed?

i. Each paw track is converted to x/y velocity magnitude with `np.gradient`, NaNs in x/y are nearest-filled before differentiation, the result is interpolated onto the neural time axis, and if both paw features are available they are averaged.

ii.

```python
if not is_tongue:
    for pos in [x_pos, y_pos]:
        mask = np.isnan(pos)
        if mask.any() and not mask.all():
            pos[mask] = np.interp(np.where(mask)[0], valid_idx, pos[valid_idx])
...
paw_vel = (paw_top + paw_bot) / 2.0
```

iii. No detailed trajectory justification was given beyond the need to handle paw features and keep the same general interpolation pipeline as other kinematics.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is discretized with the same binary session-median split used for tongue velocity, with no third `not visible` class.

ii.

```python
if paw_vel is not None:
    paw_vel_disc = discretize_per_session(paw_vel)
else:
    paw_vel_disc = np.zeros((N_TIMEBINS, n_trials), dtype=np.int32)
```

iii. The trajectory treats all three continuous movement outputs as using the same 50th-percentile discretization scheme (step 21).

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The code uses `frameTimes - vidshift - goCue` and linearly interpolates the paw-velocity trace onto `TIME_AXIS`, with synthetic frame times used as a fallback.

ii.

```python
aligned_frame_times = frame_times[:n_frames] - vidshift - go_time
interp_fn = interp1d(aligned_frame_times, vel, kind='linear', bounds_error=False, fill_value=np.nan)
velocities[:, ti] = interp_fn(TIME_AXIS)
```

iii. The agent's general plan was to align video streams with a video offset and resample them to the neural time axis (step 21).

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from the separate `motionEnergy_<session>.mat` file and the per-trial motion-energy traces under its `me/data` field.

ii.

```python
def load_motion_energy(filepath):
    mat = scipy.io.loadmat(filepath, squeeze_me=False)
    me = mat['me']
    data_field = me['data'][0, 0]
    ...
    return {'data': me_data, 'moveThresh': threshold}
```

iii. The trajectory explicitly says motion energy comes directly from the motion files and should be aligned to the neural time axis afterward (step 21).

## 9-b. How is `output` *Motion energy* processed?

i. The per-frame motion-energy trace is aligned using video timing, linearly interpolated to the neural time bins, and any missing bins are nearest-filled or set to zero before discretization.

ii.

```python
interp_fn = interp1d(aligned_frame_times, me_trial,
                     kind='linear', bounds_error=False, fill_value=np.nan)
me_interp[:, ti] = interp_fn(TIME_AXIS)
...
if mask.any() and not mask.all():
    col[mask] = np.interp(np.where(mask)[0], valid_idx, col[valid_idx])
elif mask.all():
    me_interp[:, ti] = 0.0
```

iii. Step 21 states that motion energy should be interpolated to the neural time axis and discretized with the other movement outputs.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded with the same binary 50th-percentile split as the other continuous outputs, with no `no video` category.

ii.

```python
if me_interp is not None:
    me_disc = discretize_per_session(me_interp)
else:
    me_disc = np.zeros((N_TIMEBINS, n_trials), dtype=np.int32)
```

iii. The trajectory says all three continuous outputs should use the same per-session median split (step 21).

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The script uses camera frame times from the first trajectory camera when available, subtracts `vidshift` and `goCue`, then interpolates the motion-energy trace to `TIME_AXIS`. If frame times are unavailable, it synthesizes them at 400 Hz.

ii.

```python
if len(session['traj']) > 0 and trial_idx < len(session['traj'][0]['trials']):
    ft = session['traj'][0]['trials'][trial_idx].get('frameTimes')
...
aligned_frame_times = frame_times - vidshift - go_time
interp_fn = interp1d(aligned_frame_times, me_trial, kind='linear', bounds_error=False, fill_value=np.nan)
```

iii. The trajectory framed motion energy as another video stream to be offset-corrected and resampled onto the neural grid (step 21).

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles missingness by imputing or defaulting rather than preserving an explicit missing category: missing stimulation/no-response arrays become zeros, missing `vidshift` becomes 0, missing frame times are synthesized at 400 Hz, NaN paw/motion-energy bins are nearest-filled, and all-NaN outputs are replaced by zeros. Behavior-only sessions are skipped.

ii.

```python
session['stim_enable'] = np.zeros(Ntrials)
...
session['vidshift'] = 0.0
...
if frame_times is None or np.all(np.isnan(frame_times)):
    frame_times = np.arange(1, n_frames + 1) / 400.0
...
col[mask] = np.interp(np.where(mask)[0], valid_idx, col[valid_idx])
...
elif mask.all():
    velocities[:, ti] = 0.0
```

iii. The trajectory says the agent needed to “handle missing data gracefully,” added v5/v7 fallbacks, skipped behavior-only sessions, and accepted zeros/degenerate outputs where signals were missing or sparse (steps 21, 72, 77, 93).

## 11-a. What are the most time-consuming steps of the code?

i. The heaviest work in this script is the nested per-unit, per-trial spike loop in `align_and_bin_spikes`, followed by the per-trial interpolation loops used for kinematics and motion energy. The code does not have the reference solution's vectorized spike counting path.

ii.

```python
for ui, ci in enumerate(cluster_indices):
    ...
    for ti, trial_idx in enumerate(valid_trials):
        ...
        counts, _ = np.histogram(aligned_times, bins=edges)
...
for ti, trial_idx in enumerate(valid_trials):
    ...
    interp_fn = interp1d(aligned_frame_times, vel, ...)
```

iii. The trajectory does not explicitly profile runtime; this conclusion comes from the structure of the final code and the agent's emphasis on robustness rather than vectorization.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidates are the unit-by-trial spike loop in `align_and_bin_spikes`, the per-trial NaN-filling loops, and the per-trial interpolation loops for tongue, paw, and motion energy.

ii.

```python
for ui, ci in enumerate(cluster_indices):
    for ti, trial_idx in enumerate(valid_trials):
        ...
for ti in range(n_trials):
    col = velocities[:, ti]
    ...
for ti, trial_idx in enumerate(valid_trials):
    ...
    me_interp[:, ti] = interp_fn(TIME_AXIS)
```

iii. The trajectory never addresses vectorization directly. This is an inference from the final code structure.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats similar interpolation, NaN-filling, and discretization logic across tongue, paw, and motion-energy streams. It also extracts two paw features separately and may average them after duplicating the same per-trial processing.

ii.

```python
paw_top = extract_kinematic_feature(session, 1, 'top_paw', ...)
paw_bot = extract_kinematic_feature(session, 1, 'bottom_paw', ...)
...
if tongue_vel is not None:
    tongue_vel_disc = discretize_per_session(tongue_vel)
if paw_vel is not None:
    paw_vel_disc = discretize_per_session(paw_vel)
if me_interp is not None:
    me_disc = discretize_per_session(me_interp)
```

iii. The trajectory does not discuss repetition as a design goal. This is a property of the final script.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads and stores several fields it never uses downstream (`L`, `sample`, `delay`, `no`, `moveThresh`, dates, some metadata), defines an unused `compute_velocity` helper, and constructs extra region/date bookkeeping that does not affect the actual decoder arrays.

ii.

```python
session['L'] = ...
session['sample'] = ...
session['delay'] = ...
session['no'] = ...
...
def compute_velocity(positions, dt_video=1/400):
    ...
...
return {'data': me_data, 'moveThresh': threshold}
```

iii. The trajectory focuses on getting the pipeline to run, not on trimming unused work. No explicit efficiency justification was given.
