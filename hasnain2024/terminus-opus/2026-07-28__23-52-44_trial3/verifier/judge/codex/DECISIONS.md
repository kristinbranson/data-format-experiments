# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a `SESSION_META` table with 44 sessions and, for each session, stores animal, date, probe, directory, and task. `process_session` builds `data_structure_<session>.mat`, loads it with `load_session` (auto-detecting HDF5/v7.3 versus v5 MATLAB), and separately loads `motionEnergy_<session>.mat`. It loads behavior, clusters, trajectories, and SpikeGLX metadata into one per-session dict. It only uses one probe per session, so the three `JEB15` two-probe sessions are not concatenated and several probe choices differ from the reference.

ii.
```python
SESSION_META = [
    {'anm': 'EKH1', 'date': '2021-08-07', 'probe': 2, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    ...
    {'anm': 'JEB24', 'date': '2023-11-03', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
]
```
```python
data_path = os.path.join('data', data_dir, f'data_structure_{session_id}.mat')
session_data = load_session(data_path, probe)
me_path = os.path.join('data', data_dir, f'motionEnergy_{session_id}.mat')
me = load_motion_energy(me_path) if os.path.exists(me_path) else None
```

iii. In `CONVERSION_NOTES.md`, the AI says it based session inclusion on the authors' loading scripts, included both `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`, and added support for both MATLAB formats plus three motion-energy layouts.

## 1-b. How are the data split into subjects?

i. Subjects are taken from `session_meta['anm']` and stored per processed session as `result['subject']`. The top-level `subjects` list is built in first-seen order, and `subject_idx` stores the index of each session's subject in that list.

ii.
```python
result = {
    'neural': neural_trials,
    'input': input_trials,
    'output': output_trials,
    'subject': anm,
    'session_id': session_id,
    ...
}
```
```python
if subj not in all_subjects:
    all_subjects.append(subj)
subject_idx.append(all_subjects.index(subj))
```

iii. The notes describe 14 unique animals and treat the animal/session metadata from the loading scripts as the source of subject identity.

## 1-c. How are the data split into sessions?

i. One `SESSION_META` row is one session. `main()` loops over those rows, `process_session()` converts one session at a time, and each result becomes one entry in the top-level `neural`, `input`, and `output` session lists.

ii.
```python
for sess_meta in sessions_to_process:
    result = process_session(sess_meta, PARAMS, show_processing=args.show_processing)
    if result is None:
        continue
    all_neural.append(result['neural'])
    all_input.append(result['input'])
    all_output.append(result['output'])
```

iii. The notes say the AI intentionally combined fixed-delay and randomized-delay ephys sessions into one converted dataset because both contain neural and behavior data.

## 1-d. How are the data split into trials?

i. Trials are taken directly from `ntrials` in the loaded session data. The code loops over `trial_idx in range(ntrials)` for neural and video processing, uses 1-based `clu['trial']` values to assign spikes to trials, and later keeps only `valid_trial_indices` when constructing per-trial outputs.

ii.
```python
ntrials = session_data['ntrials']
for trial_idx in range(ntrials):
    trial_num = trial_idx + 1
    spike_mask = trial_nums == trial_num
```
```python
valid_trial_indices = np.where(valid_trial_mask)[0]
for i, trial_idx in enumerate(valid_trial_indices):
    neural = trialdat[:, :, trial_idx].T.copy()
```

iii. The AI does not give a separate written justification here beyond treating the Bpod trial table and per-spike trial labels as the native trial definition.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops early-lick trials, `no` trials, and photostimulation trials, then requires each kept trial to be either a hit or a miss. After building per-trial arrays, it also removes trials whose neural matrix is all zeros.

ii.
```python
valid_trial_mask = ~session_data['early'] & ~session_data['no'] & ~session_data['stim_enable']
valid_trial_mask = valid_trial_mask & (session_data['hit'] | session_data['miss'])
valid_trial_indices = np.where(valid_trial_mask)[0]
```
```python
for ni, ii, oi in zip(neural_trials, input_trials, output_trials):
    if np.all(ni == 0):
        n_removed += 1
        continue
```

iii. The notes say the AI followed the methods statement that early-lick and ignore trials were omitted, excluded stim trials, and treated all-zero neural trials as sessions where recording had ended.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `session_data['clusters']`, specifically each cluster's `trialtm` and `trial` arrays, plus `session_data['goCue']` for alignment. The selected probe is fixed by `SESSION_META`, and cluster `quality` is used for QC.

ii.
```python
trialtm = clu['trialtm']
trial_nums = clu['trial']
goCue = session_data['goCue']
aligned_times = trialtm[spike_mask] - goCue[trial_idx]
```
```python
q = clu['quality'].lower().strip().replace('\x00', '')
if q in excluded:
    continue
```

iii. The notes explicitly tie this to the reference functions `findClusters.m` and `alignSpikes.m`.

## 2-b. How is the `neural` data processed?

i. For each kept cluster and trial, the AI subtracts the trial's go cue from spike times, bins spikes into 5 ms bins, converts counts to Hz, smooths each trial with a causal Gaussian kernel of window 15, stacks the results into `trialdat`, and later transposes each trial to `(neurons, time)`.

ii.
```python
counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts.astype(np.float32) / dt
fr_smooth = smooth_causal(fr, params['smooth_window'], bctype='none')
trialdat[:, neuron_idx, trial_idx] = fr_smooth
```
```python
def make_causal_gaussian_kernel(N):
    kern = np.array(sig_windows.gaussian(N, std=np.std(np.arange(N))))
    kern[:N // 2] = 0
    kern = kern / kern.sum()
    return kern
```

iii. In the notes, the AI says this matches `getSeq.m` and `mySmooth.m`, and explicitly states that it chose the code's causal smoothing implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It first excludes only clusters whose lower-cased quality label is in `{'garbage', 'gabrga', 'noisy', 'real?'}` and keeps empty/null labels. It then removes neurons with mean firing rate `<= 0.5 Hz`, and skips sessions with fewer than 10 remaining units.

ii.
```python
'excluded_qualities': {'garbage', 'gabrga', 'noisy', 'real?'},
'low_fr': 0.5,
'min_units': 10,
```
```python
q = clu['quality'].lower().strip().replace('\x00', '')
if q in excluded:
    continue
...
mean_fr = np.mean(np.mean(trialdat, axis=2), axis=0)
keep_mask = mean_fr > params['low_fr']
```

iii. `CONVERSION_NOTES.md` says the AI intentionally used the code default `0.5 Hz` instead of the paper's `1 Hz`, and that keeping unlabeled qualities matched `findClusters.m`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is done by subtracting the trial's go-cue time from each spike time before binning.

ii.
```python
aligned_times = trialtm[spike_mask] - goCue[trial_idx]
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The notes explicitly cite `alignSpikes.m` and describe the alignment event as `goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data use 5 ms bins from `-2.5 s` to `+2.5 s` around go cue. No further neural rebinning is applied after histogramming, although non-neural streams are interpolated onto the same 5 ms grid.

ii.
```python
'dt': 1.0 / 200.0,
'tmin': -2.5,
'tmax': 2.5,
...
edges = np.arange(tmin, tmax + dt, dt)
time_axis = edges[:-1] + dt / 2
```

iii. The notes say these values were taken from `getDefaultParams.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a raw variable directly. It is derived from the global alignment window parameters (`tmin`, `tmax`, `dt`) chosen around go cue and represented as bin centers.

ii.
```python
tmin, tmax, dt = params['tmin'], params['tmax'], params['dt']
edges = np.arange(tmin, tmax + dt, dt)
time_axis = edges[:-1] + dt / 2
```
```python
time_input = time_axis.reshape(1, -1).astype(np.float32)
```

iii. The notes describe this as the decoder input "Time from goCue onset" defined on the same 5 ms axis as the neural data.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes the centers of the 5 ms bins and reshapes them into a `(1, n_timepoints)` array for every trial.

ii.
```python
time_axis = edges[:-1] + dt / 2
time_input = time_axis.reshape(1, -1).astype(np.float32)
input_trials.append(time_input)
```

iii. No additional justification is documented beyond using the shared time grid.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same `time_axis` is used for spike binning and for the decoder input, so the input is aligned by construction to the neural bins.

ii.
```python
counts, _ = np.histogram(aligned_times, bins=edges)
...
time_input = time_axis.reshape(1, -1).astype(np.float32)
```

iii. The notes say the input axis is `[-2.5, 2.5]` with 1000 bins at 5 ms, matching the neural axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction only from `session_data['R']` on the already filtered trials. It does not combine `R` with `hit`/`miss`, and it drops `no` trials before this step.

ii.
```python
# Lick direction: R=1, L=0
lick_direction = session_data['R'][valid_trial_indices].astype(int)
```

iii. In the mapping notes, the AI explicitly planned `bp.R -> output[0] (lick_direction)`.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI simply casts the right-choice flag to `0/1` and broadcasts that value across all time bins for the trial.

ii.
```python
output = np.zeros((6, n_timepts), dtype=np.int64)
output[0, :] = lick_direction[i]
```

iii. The written plan says only `L=0, R=1, per-trial`; there is no documented attempt to recover actual lick direction on miss trials or represent no-lick trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `session_data['autowater']`.

ii.
```python
context = (session_data['autowater'][valid_trial_indices] == 0).astype(int)
```

iii. The notes explicitly map `bp.autowater` to context and describe `WC=0, DR=1`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI converts `autowater == 0` to `1` for DR and `autowater != 0` to `0` for WC, then broadcasts the result across time bins.

ii.
```python
# Context: WC=0, DR=1
context = (session_data['autowater'][valid_trial_indices] == 0).astype(int)
output[1, :] = context[i]
```

iii. The notes justify this as a direct mapping from the water-cued flag to the prompt's requested labels.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived only from `session_data['hit']` after trial filtering. `miss` only matters indirectly because the filter requires `hit | miss`, and `no` trials have already been removed.

ii.
```python
valid_trial_mask = valid_trial_mask & (session_data['hit'] | session_data['miss'])
...
outcome = session_data['hit'][valid_trial_indices].astype(int)
```

iii. The mapping notes say `bp.hit -> output[2] (outcome)`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI makes outcome binary: hit becomes `1` and anything else among the retained trials becomes `0`. Because ignore trials are filtered out, the stored classes are effectively "correct" versus "miss".

ii.
```python
# Outcome: incorrect=0, correct=1
outcome = session_data['hit'][valid_trial_indices].astype(int)
output[2, :] = outcome[i]
```

iii. The notes justify this with the target label request `incorrect=0, correct=1` and the earlier decision to keep only hit/miss trials.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-camera trajectory entry `session_data['traj'][0]`, specifically the `tongue` feature in `ts`, plus `frameTimes`, `goCue`, and the session video offset. The code does not use the bottom-camera tongue feature.

ii.
```python
cam0 = session_data['traj'][0]  # side cam
...
if fn.lower() == 'tongue':
    tongue_idx = fi
```
```python
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_ft, y, kind='linear', bounds_error=False, fill_value=np.nan)
```

iii. The notes say only "DLC tongue velocity" plus "fill NaN positions with baseline, set NaN velocity to 0"; they do not mention combining the two tongue views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the AI interpolates side-camera tongue `x` and `y` positions directly onto the 5 ms neural grid, fills missing positions with the session-wide mean position, takes simple gradients of the filled traces, zeros those gradients where the original position was missing, and uses the speed magnitude as the continuous tongue signal.

ii.
```python
all_x_filled[np.isnan(all_x_filled)] = mean_x
all_y_filled[np.isnan(all_y_filled)] = mean_y
...
xvel = np.gradient(all_x_filled[:, trial_idx])
yvel = np.gradient(all_y_filled[:, trial_idx])
nan_mask = np.isnan(all_x[:, trial_idx])
xvel[nan_mask] = 0.0
yvel[nan_mask] = 0.0
speed = np.sqrt(xvel**2 + yvel**2)
```

iii. The notes explicitly justify baseline-filling with "Following setTongueBaselinePosition" and, in the trajectory, the AI says it set invisible-tongue velocity to zero and changed thresholding to avoid an all-ones output when the median was zero.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI thresholds the session's tongue velocities at the 50th percentile of non-NaN values. If the threshold equals the minimum value, it uses `>` instead of `>=` to avoid all bins becoming class 1. NaNs are converted to class 0.

ii.
```python
threshold = np.percentile(valid_vals, 50)
if threshold <= np.min(valid_vals) + 1e-10:
    discretized = (vel_data > threshold).astype(int)
else:
    discretized = (vel_data >= threshold).astype(int)
discretized[np.isnan(vel_data)] = 0
```

iii. The trajectory explicitly records this as a fix for the case where mostly zero tongue velocities made the median threshold zero and produced all-ones output.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI aligns tongue positions by subtracting the session video offset and the trial's go cue from `frameTimes`, then interpolates those aligned coordinates onto the same `time_axis` used by the neural data.

ii.
```python
vidshift = find_video_offset(session_data)
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
all_x[:, trial_idx] = fx(time_axis)
```

iii. The notes say the video offset computation matches `findVideoOffset.m` and that kinematic outputs are aligned to neural time.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `session_data['traj'][1]`, using the `top_paw` feature when present and otherwise falling back to the first feature name containing `paw`. It also uses `frameTimes`, `goCue`, and the session video offset.

ii.
```python
cam1 = session_data['traj'][1]  # top cam
...
if 'top_paw' in fn.lower():
    paw_idx = fi
...
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
```

iii. The notes justify this as "Use top_paw from top camera" and call it the reliable paw view.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI interpolates paw positions directly onto the neural time grid, linearly fills missing values inside that grid, takes simple gradients, subtracts each trial's median derivative as a baseline, optionally fills NaNs in the derivatives, and computes speed magnitude.

ii.
```python
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_ft, y, kind='linear', bounds_error=False, fill_value=np.nan)
x_interp = fx(time_axis)
y_interp = fy(time_axis)
...
xvel = np.gradient(x_interp)
yvel = np.gradient(y_interp)
basederiv_x = np.nanmedian(np.diff(x_interp))
basederiv_y = np.nanmedian(np.diff(y_interp))
xvel -= basederiv_x
yvel -= basederiv_y
speed = np.sqrt(xvel**2 + yvel**2)
```

iii. The notes explicitly say "subtract baseline derivative" for paw velocity.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It uses the same `discretize_velocity()` function as tongue velocity: session median split on non-NaN values, `>` instead of `>=` when the threshold equals the minimum, and NaNs mapped to 0.

ii.
```python
paw_disc, paw_thresh = discretize_velocity(valid_paw_vel)
```
```python
if threshold <= np.min(valid_vals) + 1e-10:
    discretized = (vel_data > threshold).astype(int)
else:
    discretized = (vel_data >= threshold).astype(int)
```

iii. The notes say all three continuous outputs use per-session 50th-percentile thresholds.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw positions are aligned the same way as tongue positions: `frameTimes - vidshift - goCue`, then interpolation onto the shared neural `time_axis`.

ii.
```python
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
x_interp = fx(time_axis)
```

iii. The notes do not give a paw-specific justification beyond using the same video offset correction as the other video features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<session>.mat` file, specifically `me['data']` after format unwrapping. For alignment it also uses side-camera `frameTimes`, `goCue`, `bitStart`, `sglx_bitstart`, and `sglx_fs`.

ii.
```python
d = sio.loadmat(me_filepath, squeeze_me=False)
me_raw = d['me']
...
return {'data': trials, 'moveThresh': me_thresh}
```
```python
ft = cam0['trials'][trial_idx]['frameTimes']
aligned_ft = ft - vidshift - goCue[trial_idx]
```

iii. The notes say the AI intentionally loaded motion energy from the standalone files and added support for three file layouts.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI interpolates each trial's motion-energy trace onto the neural time grid. If trace length matches `frameTimes`, it interpolates directly from aligned frame times; otherwise it assumes a 400 Hz trace, builds a fallback aligned time vector, interpolates from that, and then linearly fills remaining NaNs within each trial.

ii.
```python
if len(trial_me) == len(ft):
    f_interp = interp1d(aligned_ft, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
    me_aligned[:, trial_idx] = f_interp(time_axis)
else:
    me_times = np.arange(len(trial_me)) / 400.0
    aligned_me_times = me_times - 0.5 - goCue[trial_idx] + session_data['bitStart'][trial_idx]
    f_interp = interp1d(aligned_me_times, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
```
```python
if np.any(np.isnan(col)) and np.any(~np.isnan(col)):
    col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), col[not_nans])
```

iii. The mapping notes say motion energy is "Interpolated to neural time" and then median-thresholded.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. It is discretized with the same helper as tongue and paw: session 50th percentile threshold on non-NaN values, strict `>` at the minimum edge case, and NaNs mapped to 0.

ii.
```python
me_disc, me_thresh = discretize_velocity(valid_me)
```

iii. The notes say the AI used per-session 50th-percentile thresholds for all continuous outputs.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned either with side-camera `frameTimes - vidshift - goCue` or, in the mismatch fallback branch, with a synthetic 400 Hz time base shifted by `-0.5 - goCue + bitStart`, and then interpolated onto the neural `time_axis`.

ii.
```python
aligned_ft = ft - vidshift - goCue[trial_idx]
f_interp = interp1d(aligned_ft, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
```
```python
aligned_me_times = me_times - 0.5 - goCue[trial_idx] + session_data['bitStart'][trial_idx]
f_interp = interp1d(aligned_me_times, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
```

iii. The notes justify this at a high level by saying the code matches `loadMotionEnergy.m` and uses the same video-offset correction as the video streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses broad fallbacks rather than preserving missingness. Missing or absent fields are often replaced with empty arrays or zeros; missing video offset falls back to `0.0`; motion-energy NaNs are linearly filled; missing tongue positions are replaced with the session mean and their velocities zeroed; missing paw positions and derivatives are interpolated; empty motion-energy loads return `None`; all-zero neural trials are dropped after construction.

ii.
```python
except:
    data['stim_enable'] = np.zeros(ntrials, dtype=bool)
...
except:
    vidshift = 0.0
```
```python
all_x_filled[np.isnan(all_x_filled)] = mean_x
all_y_filled[np.isnan(all_y_filled)] = mean_y
...
col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), col[not_nans])
```

iii. The notes justify this as handling multiple MATLAB layouts, null quality labels, and edge cases like end-of-session all-zero trials; the trajectory also shows the AI deliberately changing thresholding because invisible tongue frames produced mostly zero velocities.

## 11-a. What are the most time-consuming steps of the code?

i. The code's heaviest work is the nested `for neuron_idx` × `for trial_idx` spike binning/smoothing loop, plus the per-trial interpolation and gradient computation for tongue, paw, and motion energy. Session loading is also substantial, and the notes report about 8 seconds per session and about 247 seconds for the full run.

ii.
```python
for neuron_idx, clu in enumerate(valid_clusters):
    ...
    for trial_idx in range(ntrials):
        ...
        counts, _ = np.histogram(aligned_times, bins=edges)
        fr_smooth = smooth_causal(fr, params['smooth_window'], bctype='none')
```
```python
for trial_idx in range(min(ntrials, len(cam0['trials']))):
    ...
    fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
```

iii. The notes only justify this indirectly with runtime measurements; they do not provide a deeper performance analysis.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization candidates are the neuron-by-trial spike loop, the per-trial motion-energy interpolation/fill loop, the per-trial tongue and paw interpolation/gradient loops, and some repeated one-array-at-a-time `np.interp` filling loops.

ii.
```python
for neuron_idx, clu in enumerate(valid_clusters):
    ...
    for trial_idx in range(ntrials):
```
```python
for trial_idx in range(ntrials):
    col = me_aligned[:, trial_idx]
    ...
for arr in [x_interp, y_interp]:
    ...
```

iii. The AI does not document an explicit reason for leaving these as loops.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several computations: it rebuilds `time_input` for every trial even though it is session-constant; it interpolates each stream independently trial by trial; it runs separate missing-data fill loops on many arrays; and it initializes `brain_region_idx` once in the output dict and then rebuilds it again correctly afterward.

ii.
```python
for i, trial_idx in enumerate(valid_trial_indices):
    ...
    time_input = time_axis.reshape(1, -1).astype(np.float32)
    input_trials.append(time_input)
```
```python
'brain_region_idx': [np.zeros(len(sess_neural), dtype=int) for sess_neural in all_neural],
...
data['brain_region_idx'] = []
for sess_neural in all_neural:
    ...
```

iii. There is no explicit written justification for these repeated steps.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The loader reads many fields that are not used in the final conversion, including `sample`, `delay`, `lickL`, `lickR`, `L`, and motion-energy `moveThresh`. It also creates an incorrect first pass of `brain_region_idx` and overwrites it, and the optional plotting block produces diagnostics not used downstream.

ii.
```python
data['lickL'] = []
data['lickR'] = []
data['sample'] = ev['sample'][0, :]
data['delay'] = ev['delay'][0, :]
```
```python
return {'data': trials, 'moveThresh': me_thresh}
```
```python
'brain_region_idx': [np.zeros(len(sess_neural), dtype=int) for sess_neural in all_neural],
...
data['brain_region_idx'] = []
```

iii. The notes do not defend these extra reads or the duplicate `brain_region_idx` construction; they only mention the optional processing plots as a validation aid.
