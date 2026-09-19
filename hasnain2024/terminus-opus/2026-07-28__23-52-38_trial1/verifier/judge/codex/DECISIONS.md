# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent auto-discovers sessions by scanning `data/Ephys_Behavior` and `data/RandomizedDelay_Ephys_Behavior` for every `data_structure_*.mat` file, while separately parsing MATLAB loading scripts only to recover per-session probe numbers. It then loads each session with an HDF5-or-v5 auto-detecting loader and loads motion energy from a separate `motionEnergy_*.mat` file when present.

ii.
```python
def get_available_sessions():
    sessions = []
    script_dir = 'code/DataLoadingScripts/Recording and video'
    probe_map = {}
    ...
    for data_dir in ['data/Ephys_Behavior', 'data/RandomizedDelay_Ephys_Behavior']:
        ...
        for fn in sorted(os.listdir(data_dir)):
            if not fn.startswith('data_structure_'):
                continue
            ...
            probes = probe_map.get((animal, date), [1])
            ...
            sessions.append({
                'animal': animal,
                'date': date,
                'filepath': filepath,
                'probes': probes,
                'me_filepath': me_filepath if has_me else None,
                'data_dir': data_dir
            })
```
```python
def load_session_data(filepath):
    fmt = detect_mat_format(filepath)
    if fmt == 'hdf5':
        return load_session_data_hdf5(filepath)
    elif fmt == 'v5':
        return load_session_data_v5(filepath)
```

iii. In `CONVERSION_NOTES.md`, the agent says the dataset contains 47 total ephys sessions across the two folders and describes the dual-format loader as necessary because some files are MATLAB v7.3 HDF5 and others are v5. The trajectory shows it deliberately compared loading scripts against on-disk files, then chose to include all available ephys files rather than transcribing the authors' exact session list.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the animal prefix in each discovered filename, stored as `animal`, then converted into a sorted unique `subjects` list with `subject_idx` pointing from each included session into that list.

ii.
```python
sessions.append({
    'animal': animal,
    'date': date,
    ...
})
```
```python
subjects = sorted(set(r['animal'] for r in results))
subject_idx = np.array([subjects.index(r['animal']) for r in results])
```

iii. The notes repeatedly describe sessions as `ANIMAL_DATE` and report subject counts by animal code. The trajectory also shows the agent identified animals by parsing filenames and loading scripts rather than relying on inconsistent in-file metadata.

## 1-c. How are the data split into sessions?

i. One session is one discovered `data_structure_<animal>_<date>.mat` file. Sessions from the two ephys folders are pooled together, then only sessions that pass the later inclusion checks and do not error out are kept in the final dataset.

ii.
```python
for data_dir in ['data/Ephys_Behavior', 'data/RandomizedDelay_Ephys_Behavior']:
    ...
    for fn in sorted(os.listdir(data_dir)):
        if not fn.startswith('data_structure_'):
            continue
        ...
        sessions.append({...})
```
```python
for i, sess_info in enumerate(all_sessions):
    ...
    result = process_session(sess_info, time_axis, show_processing=args.show_processing)
    if result is not None:
        results.append(result)
```

iii. In the notes the agent explicitly reports "Found 47 total sessions" and later "43 sessions included." Its justification is that all available ephys session files should be considered, with behavior-only MAH sessions skipped and later curation deciding what remains.

## 1-d. How are the data split into trials?

i. Trials are indexed from `bp['Ntrials']` and processed as integer trial numbers `0..Ntrials-1`. Neural data use `unit['trial']` and `unit['trialtm']` to assign spikes to each trial. Behavioral outputs and movement variables are then gathered by iterating over the final `trial_indices`.

ii.
```python
bp['Ntrials'] = int(get_field(bp_raw, 'Ntrials').flatten()[0])
...
for j in range(Ntrials):
    trial_num = j + 1
    spk_mask = trial_nums == trial_num
```
```python
trial_indices = np.where(valid_trials)[0]
...
for trial_idx in trial_indices:
    neural = firing_rates[:, :, trial_idx].astype(np.float32)
    ...
    output_trials.append(out)
```

iii. The notes describe `obj.bp` as trial-structured and `obj.clu`/`obj.traj` as carrying trial-wise information. The agent's trajectory shows it explored `obj.bp/Ntrials`, `obj/clu`, and `obj/traj` and then treated `Ntrials` as the authoritative trial count.

## 1-e. How are trials filtered based on quality controls?

i. The code keeps only trials that are not early licks and not stimulation trials, and then additionally drops all `bp['no']` ignore trials. At the session level it also excludes any session without more than 40 right-hit DR trials and 40 left-hit DR trials, and excludes sessions with fewer than 10 kept units or fewer than 2 valid trials. It does not remove late behavioral trials that occur after neural recording has effectively ended, which is why the verification log contains all-zero neural trials.

ii.
```python
def check_session_inclusion(session_data, probes):
    r_hit_dr = bp['R'] & bp['hit'] & ~bp['stim_enable'] & ~bp['autowater'] & ~bp['early']
    l_hit_dr = bp['L'] & bp['hit'] & ~bp['stim_enable'] & ~bp['autowater'] & ~bp['early']
    return n_r > MIN_HIT_TRIALS and n_l > MIN_HIT_TRIALS, n_r, n_l
```
```python
valid_trials = ~bp['early'] & ~bp['stim_enable']
valid_trials = valid_trials & ~bp['no']
trial_indices = np.where(valid_trials)[0]
...
if n_neurons < MIN_UNITS:
    return None
```

iii. `CONVERSION_NOTES.md` says the agent matched `UseInclusionCritera.m`, excluded early/stim trials, and required at least 10 units. Later notes acknowledge the resulting dataset still has all-zero neural trials in two sessions and says the decoder should "handle these gracefully," rather than dropping them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from `obj.clu` spike-sorted unit fields, especially `trial`, `trialtm`, and `quality`, together with `bp.ev.goCue` for alignment.

ii.
```python
unit['trial'] = unit_raw['trial'].flatten().astype(int)
unit['trialtm'] = unit_raw['trialtm'].flatten().astype(float)
unit['quality'] = str(q_flat[0]).strip()
...
goCue = session['bp']['ev']['goCue']
```

iii. The notes identify `obj.clu` as the spike source, `alignSpikes.m` as subtracting `goCue`, and cluster quality as part of curation. The trajectory shows the agent explicitly inspected these fields before writing the conversion.

## 2-b. How is the `neural` data processed?

i. For each kept unit and each trial, spikes are aligned to go cue, histogrammed into 5 ms bins, converted to firing rate by dividing by `DT`, and smoothed with a causal Gaussian-like kernel of length 15 bins using `np.convolve(..., mode='same')`. Units from multiple probes are concatenated afterward. No baseline subtraction or z-scoring is applied.

ii.
```python
DT = 1.0 / 200.0
SMOOTH_N = 15
...
counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts.astype(np.float64) / DT
fr_smooth = smooth_signal(fr, SMOOTH_N)
firing_rates[i, :, j] = fr_smooth
```
```python
def causal_gaussian_kernel(N):
    kern = np.array(sig_windows.gaussian(N, std=np.std(np.arange(1, N+1))))
    kern[:N//2] = 0
    kern = kern / kern.sum()
    return kern
```

iii. The notes say this matches `getSeq.m` and `mySmooth.m`, and the trajectory explicitly records the agent's conclusion that the reference code uses `smooth=15` with a causal Gaussian window. That is the stated reason for choosing this smoothing instead of a symmetric filter.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered in two stages. First, only clusters whose quality string does not contain `"garbage"` are kept. Second, units with mean firing rate `<= 0.5 Hz` across all times and trials are removed. Sessions with fewer than 10 remaining units are then excluded entirely.

ii.
```python
LOW_FR = 0.5
MIN_UNITS = 10
...
for u_idx, unit in enumerate(units):
    quality = unit['quality'].lower().strip()
    if 'garbage' not in quality:
        valid_units.append(u_idx)
...
mean_frs = np.mean(np.mean(firing_rates, axis=2), axis=1)
fr_mask = mean_frs > LOW_FR
...
if n_neurons < MIN_UNITS:
    return None
```

iii. In `CONVERSION_NOTES.md`, the agent cites `getDefaultParams.m` and `deleteGarbageClu.m` as justification for `lowFR=0.5` and removing only garbage-labeled clusters. It also states it matched session inclusion criteria requiring at least 10 units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting the per-trial go cue time from each spike's `trialtm` within that trial.

ii.
```python
aligned_times = trialtm[spk_mask] - goCue[j]
```

iii. The notes and trajectory explicitly cite `alignSpikes.m` and summarize its rule as `trialtm_aligned = trialtm - goCue(trial)`, which is what the code implements.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use `DT = 1/200 = 5 ms`. The agent builds `time_axis` with `np.arange(TMIN, TMAX + DT/2, DT)`, giving 1001 samples from `-2.5` to `2.5` inclusive, then bins spikes with edges formed by appending one more edge at `time_axis[-1] + DT`. There is no later temporal rebinning.

ii.
```python
DT = 1.0 / 200.0  # 5ms time bins
TMIN = -2.5
TMAX = 2.5
...
time_axis = np.arange(TMIN, TMAX + DT/2, DT)
time_axis = time_axis[time_axis <= TMAX + 1e-10]
...
edges = np.append(time_axis, time_axis[-1] + DT)
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The notes justify 5 ms bins by citing `dt=1/200` from the MATLAB code. They also state, incorrectly relative to the final pickle shape, that this "matches" the reference time axis.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a dedicated raw variable. It is a synthetic time grid defined by `TMIN`, `TMAX`, and `DT`, intended to represent seconds from the aligned go cue.

ii.
```python
def make_time_axis():
    time_axis = np.arange(TMIN, TMAX + DT/2, DT)
    time_axis = time_axis[time_axis <= TMAX + 1e-10]
    return time_axis
```
```python
inp = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes map "time from goCue" directly to "Time axis -2.5 to 2.5" and treat it as a decoder-required constructed variable rather than a field from the raw files.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The agent creates an evenly spaced time vector from `-2.5` to `2.5` seconds in 5 ms steps and reuses that same vector for every trial and every session.

ii.
```python
time_axis = np.arange(TMIN, TMAX + DT/2, DT)
time_axis = time_axis[time_axis <= TMAX + 1e-10]
...
inp = time_axis.astype(np.float32).reshape(1, -1)
input_trials.append(inp)
```

iii. The notes say the time axis should match the reference alignment window and bin size, and the trajectory shows the agent deliberately chose MATLAB-style `tmin:dt:tmax` semantics.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same `time_axis` is used as the decoder input for every trial and is also reused when forming spike histogram edges and when interpolating video-derived outputs, so the input is meant to be the common time base for all streams.

ii.
```python
time_axis = make_time_axis()
...
edges = np.append(time_axis, time_axis[-1] + DT)
...
inp = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes explicitly say the input is "time from goCue as continuous variable" and the code structure shows that the same grid is treated as the master alignment axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The code derives lick direction only from the per-trial instruction flags `bp['R']` and implicitly `bp['L']`, after having already removed ignore (`bp['no']`) trials. It does not use `hit` and `miss` to infer actual lick direction on error trials.

ii.
```python
valid_trials = valid_trials & ~bp['no']
...
lick_dir = 1 if bp['R'][trial_idx] else 0
```

iii. In the notes' variable mapping, the agent states `obj.bp.R -> output[0]: lick_direction | R=1, L=0 | Per trial`, showing that it intentionally treated lick direction as the instructed side rather than inferred behavior.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Lick direction is binarized as `1` for right-instruction trials and `0` otherwise, then repeated across all time bins of that trial. There is no third "none/no lick" class because ignore trials were discarded earlier.

ii.
```python
lick_dir = 1 if bp['R'][trial_idx] else 0
...
out = np.zeros((6, len(time_axis)), dtype=np.int64)
out[0, :] = lick_dir
```
```python
'output_values': [
    ['left', 'right'],
    ...
]
```

iii. The notes justify this by mapping left/right trial identity directly to decoder output. No separate code-path or notes-based justification was given for miss trials or no-lick trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `bp['autowater']` flag.

ii.
```python
bp['autowater'] = get_field(bp_raw, 'autowater').flatten().astype(bool)
...
context = 0 if bp['autowater'][trial_idx] else 1
```

iii. The notes explicitly map `obj.bp.autowater` to behavioral context and describe `autowater=1` as the WC context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code relabels `autowater=True` as `WC=0` and `autowater=False` as `DR=1`, then repeats that scalar across all time bins in the trial.

ii.
```python
context = 0 if bp['autowater'][trial_idx] else 1
...
out[1, :] = context
```
```python
'output_values': [
    ...,
    ['WC', 'DR'],
    ...
]
```

iii. `CONVERSION_NOTES.md` gives exactly this mapping in Step 5.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is effectively derived from `bp['hit']` alone after ignore trials have already been removed. The code does not read `bp['miss']` when constructing the final output variable.

ii.
```python
bp['hit'] = get_field(bp_raw, 'hit').flatten().astype(bool)
bp['miss'] = get_field(bp_raw, 'miss').flatten().astype(bool)
bp['no'] = get_field(bp_raw, 'no').flatten().astype(bool)
...
valid_trials = valid_trials & ~bp['no']
...
outcome = 1 if bp['hit'][trial_idx] else 0
```

iii. The notes' variable mapping says `obj.bp.hit -> output[2]: outcome | incorrect=0, correct=1`, which matches the code's decision to encode outcome as a binary hit-vs-not-hit variable on non-ignore trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is encoded as `1` on hit trials and `0` otherwise, then repeated across all time bins. There is no explicit incorrect/correct/ignore three-way coding.

ii.
```python
outcome = 1 if bp['hit'][trial_idx] else 0
...
out[2, :] = outcome
```
```python
'output_values': [
    ...,
    ['incorrect', 'correct'],
    ...
]
```

iii. The notes justify this as a direct mapping from `hit` to correctness and do not preserve the ignored trials as a third class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj` for camera 0 only, using the feature named `'tongue'`, together with that trial's `frameTimes`, `goCue`, and the session-wide video offset from `sglx.bitcode.bitstart` and `bp.ev.bitStart`.

ii.
```python
vidshift = compute_video_offset(session)
...
tongue_vel = extract_feature_velocity(session, 0, 'tongue', time_axis, vidshift)
```
```python
feat_names = cam_data['feat_names']
...
x = ts[feat_idx, 0, :]
y = ts[feat_idx, 1, :]
aligned_times = frame_times - vidshift - goCue[trix]
```

iii. The notes say velocity computation should match the video kinematics functions and identify `obj.traj` plus video offset correction as the source. The code shows the agent chose the side-view `tongue` feature only, rather than combining both tongue views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the side-camera tongue position is linearly interpolated from frame times onto the neural time axis. Velocity is then computed as the magnitude of `np.gradient` on interpolated x and y. For tongue specifically, NaNs in the gradients are replaced by zero before the magnitude is taken. There is no explicit likelihood threshold, no per-run smoothing, and no combination with the bottom camera's tongue feature.

ii.
```python
f_x = interp1d(aligned_times[valid], x[valid], kind='linear',
              bounds_error=False, fill_value=np.nan)
f_y = interp1d(aligned_times[valid], y[valid], kind='linear',
              bounds_error=False, fill_value=np.nan)

xpos[:, trix] = f_x(time_axis)
ypos[:, trix] = f_y(time_axis)
```
```python
xv = np.gradient(xpos[:, trix])
yv = np.gradient(ypos[:, trix])
...
xv[np.isnan(xv)] = 0
yv[np.isnan(yv)] = 0
...
vel_mag = np.sqrt(xvel**2 + yvel**2)
```

iii. The notes claim the code matches `findVelocity.m`, but the concrete justification visible in the code is simpler: interpolate coordinates to the neural grid, then differentiate there. No written note justifies ignoring the second tongue view or turning invisibility into zero velocity.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Tongue velocity is split at the within-session 50th percentile computed over all non-NaN values from valid trials. Values below threshold become `0`, values at or above threshold become `1`, and NaNs are forced into class `0`; there is no separate `2 = not visible` class.

ii.
```python
tongue_valid = tongue_vel[:, trial_indices]
tongue_thresh = np.nanpercentile(tongue_valid[~np.isnan(tongue_valid)], 50) if np.any(~np.isnan(tongue_valid)) else 0
...
tv_disc = np.zeros(len(time_axis), dtype=np.int64)
tv_disc[tv >= tongue_thresh] = 1
tv_disc[np.isnan(tv)] = 0
```

iii. The notes mention a 50th-percentile discretization for tongue velocity but do not discuss the required `not visible` category. The final `output_values` also expose only `['low', 'high']`.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are shifted by the session-wide video offset and by trial-specific `goCue`, then the tongue positions are linearly interpolated directly onto the same `time_axis` used by the neural data. Alignment is therefore by interpolation to neural sample times, not by averaging available video frames within neural bins.

ii.
```python
vidshift = compute_video_offset(session)
...
aligned_times = frame_times - vidshift - goCue[trix]
...
xpos[:, trix] = f_x(time_axis)
ypos[:, trix] = f_y(time_axis)
```

iii. The notes cite `findVideoOffset.m` as justification for the clock correction and broadly describe matching the neural time axis. They do not explain why interpolation was chosen over frame binning.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj` camera 1 using the feature `'top_paw'`, together with `frameTimes`, `goCue`, and the same session-level video offset.

ii.
```python
paw_vel = extract_feature_velocity(session, 1, 'top_paw', time_axis, vidshift)
```
```python
x = ts[feat_idx, 0, :]
y = ts[feat_idx, 1, :]
aligned_times = frame_times - vidshift - goCue[trix]
```

iii. The notes' mapping lists paw velocity as a time-varying output derived from the DLC trajectories. The code shows the agent chose the `top_paw` feature only.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw x/y coordinates are linearly interpolated to the neural time axis, nearest-filled through missing samples, differentiated with `np.gradient`, baseline-shift corrected by subtracting the median finite difference, nearest-filled again, and then combined into a speed magnitude.

ii.
```python
f_x = interp1d(aligned_times[valid], x[valid], kind='linear',
              bounds_error=False, fill_value=np.nan)
...
xpos[:, trix] = f_x(time_axis)
ypos[:, trix] = f_y(time_axis)
...
if not is_tongue:
    xpos[:, trix] = _fill_nearest(xpos[:, trix])
    ypos[:, trix] = _fill_nearest(ypos[:, trix])
```
```python
xv = np.gradient(xpos[:, trix])
yv = np.gradient(ypos[:, trix])
base_x = np.nanmedian(np.diff(xpos[:, trix]))
base_y = np.nanmedian(np.diff(ypos[:, trix]))
xv = xv - base_x
yv = yv - base_y
xv = _fill_nearest(xv)
yv = _fill_nearest(yv)
vel_mag = np.sqrt(xvel**2 + yvel**2)
```

iii. The notes say the code matches `findPosition` and `findVelocity`, which is the stated justification for interpolation and baseline subtraction. There is no separate justification for not using per-frame likelihood filtering or per-run smoothing.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is split at the session 50th percentile over all finite valid-trial samples. The code emits only two classes, `0` and `1`, and maps NaNs to `0` instead of a third `not visible` class.

ii.
```python
paw_valid = paw_vel[:, trial_indices]
paw_thresh = np.nanpercentile(paw_valid[~np.isnan(paw_valid)], 50) if np.any(~np.isnan(paw_valid)) else 0
...
pv_disc = np.zeros(len(time_axis), dtype=np.int64)
pv_disc[pv >= paw_thresh] = 1
pv_disc[np.isnan(pv)] = 0
```

iii. The notes mention a 50th-percentile split for paw velocity, but, as with tongue velocity, they do not preserve the prompt's visibility class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw positions are aligned by subtracting the video offset and go cue from frame times, then linearly interpolating those positions to the neural `time_axis` before differentiation and discretization.

ii.
```python
aligned_times = frame_times - vidshift - goCue[trix]
...
xpos[:, trix] = f_x(time_axis)
ypos[:, trix] = f_y(time_axis)
```

iii. The notes justify using `findVideoOffset.m`-style correction and a shared neural time base for all outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_*.mat` file for each session. The code unwraps several possible MATLAB layouts into a per-trial list of 1D motion energy traces.

ii.
```python
def load_motion_energy(me_filepath):
    data = sio.loadmat(me_filepath, squeeze_me=False)
    me_raw = data['me']
    ...
    return {'data': me_trials, 'moveThresh': me_thresh}
```
```python
if sess_info['me_filepath'] is not None:
    me_data = load_motion_energy(sess_info['me_filepath'])
```

iii. The notes explicitly identify motion energy as a separate file and state that multiple MATLAB layouts had to be handled, which is why the loader unwraps nested `me.data` structures.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each trial, motion energy is aligned using camera 0 frame times, linearly interpolated to the neural time axis, and any remaining NaNs are nearest-filled. The resulting continuous trace is later thresholded at the session median.

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

iii. The notes say the code should align motion energy to the neural axis and handle variant file formats. They do not justify the nearest-fill step; that choice is only visible in the implementation.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is median-split within each session over all finite samples from valid trials. The code outputs only `0` and `1`, with NaNs forced to `0`; it does not emit the required `2 = no video` class.

ii.
```python
me_valid = me_aligned[:, trial_indices]
me_thresh = np.nanpercentile(me_valid[~np.isnan(me_valid)], 50) if np.any(~np.isnan(me_valid)) else 0
...
me_disc = np.zeros(len(time_axis), dtype=np.int64)
me_disc[me >= me_thresh] = 1
me_disc[np.isnan(me)] = 0
```

iii. The notes mention 50th-percentile discretization for motion energy but, like the other movement outputs, describe only `low/high` categories in the saved metadata.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera 0 frame times are shifted by the session video offset and trial go cue, then the per-frame motion energy trace is linearly interpolated onto the same `time_axis` used by neural data.

ii.
```python
cam_data = session['traj'][0]
...
aligned_times = frame_times - vidshift - goCue[trix]
...
me_aligned[:, trix] = f_me(time_axis)
```

iii. The notes justify using the reference video-offset computation and a shared neural-aligned time base, but do not justify interpolation versus frame binning.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code generally fills or defaults rather than preserving explicit missingness. Missing `reward` and `bitStart` become all-NaN arrays. Missing `sglx` yields zero video offset. Missing frame times trigger synthetic frame times at `1/VIDEO_FS`, and missing points in paw and motion-energy traces are nearest-filled. Tongue NaNs are turned into zero velocity and then below-threshold class. Missing motion-energy files yield all-NaN traces that are then thresholded into zeros. Invalid or missing `clu` structures can cause whole-session errors or skipped probes.

ii.
```python
except:
    bp['ev']['reward'] = np.full(bp['Ntrials'], np.nan)
...
if session['sglx'] is None:
    return 0.0
```
```python
if trial['frameTimes'] is not None and not np.all(np.isnan(trial['frameTimes'])):
    frame_times = trial['frameTimes']
else:
    n_frames = ts.shape[2]
    frame_times = np.arange(1, n_frames + 1) / VIDEO_FS
```
```python
def _fill_nearest(arr):
    ...
    for i in range(len(arr)):
        if mask[i]:
            dists = np.abs(valid - i)
            arr[i] = arr[valid[np.argmin(dists)]]
```
```python
xv[np.isnan(xv)] = 0
yv[np.isnan(yv)] = 0
...
me_disc[np.isnan(me)] = 0
```

iii. The notes justify "handling edge cases" such as invalid probes, dual MATLAB formats, and missing motion-energy thresholds. They also acknowledge warnings about all-zero neural trials but chose not to filter them out.

## 11-a. What are the most time-consuming steps of the code?

i. The code is instrumented so that session loading, spike processing, tongue velocity extraction, paw velocity extraction, and motion-energy alignment are timed separately. In practice the most expensive work is loading the large `.mat` files and the nested per-unit/per-trial spike processing and per-trial video interpolation loops.

ii.
```python
print(f'    Loaded in {t_load:.1f}s')
...
print(f'    Spike processing: {t_spk:.1f}s, {n_neurons} total units')
...
print(f'    Tongue velocity: {t_tongue:.1f}s')
print(f'    Paw velocity: {t_paw:.1f}s')
print(f'    Motion energy: {t_me:.1f}s')
```
```python
for i, u_idx in enumerate(valid_units):
    ...
    for j in range(Ntrials):
        ...
```

iii. The notes estimate roughly 8 seconds per session and about 5 minutes for the full run, which is the agent's main explicit justification for where time is being spent.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious vectorization targets are the nested spike loops over units and trials, the per-trial interpolation loops in `extract_feature_velocity` and `align_motion_energy`, and the `for i in range(len(arr))` nearest-fill routine. The agent left all of these as Python loops.

ii.
```python
for i, u_idx in enumerate(valid_units):
    ...
    for j in range(Ntrials):
        ...
```
```python
for trix in range(Ntrials):
    ...
    xpos[:, trix] = f_x(time_axis)
```
```python
for i in range(len(arr)):
    if mask[i]:
        dists = np.abs(valid - i)
        arr[i] = arr[valid[np.argmin(dists)]]
```

iii. The agent did not explicitly justify leaving these loops unvectorized. The only implicit justification is pragmatism: the notes focus on matching reference logic rather than optimizing runtime.

## 11-c. What processing does the code repeat multiple times?

i. The code repeatedly constructs the same per-trial time input array, repeatedly interpolates each video-derived feature trial-by-trial onto the same `time_axis`, and repeatedly nearest-fills missing values after those interpolations. It also computes separate interpolation passes for tongue, paw, and motion energy even though they share the same offset and time grid.

ii.
```python
for trial_idx in trial_indices:
    inp = time_axis.astype(np.float32).reshape(1, -1)
    input_trials.append(inp)
```
```python
tongue_vel = extract_feature_velocity(session, 0, 'tongue', time_axis, vidshift)
paw_vel = extract_feature_velocity(session, 1, 'top_paw', time_axis, vidshift)
...
me_aligned = align_motion_energy(me_data, session, time_axis, vidshift)
```

iii. There is no explicit justification for these repetitions in the notes. They are a byproduct of the agent's decision to align each signal independently to the neural time grid.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The loaders materialize many raw fields that never enter the saved output, including `tm`, `sample`, `delay`, `reward`, `NdroppedFrames`, and entire trajectory structures for unused features. The code also computes continuous interpolated x/y positions and continuous velocities only to discard them after converting to binary movement classes.

ii.
```python
unit['tm'] = unit_raw['tm'].flatten().astype(float)
...
bp['ev']['sample'] = get_field(ev_raw, 'sample').flatten().astype(float)
bp['ev']['delay'] = get_field(ev_raw, 'delay').flatten().astype(float)
...
trial_data['NdroppedFrames'] = float(nd.flatten()[0])
```
```python
xpos = np.full((len(time_axis), Ntrials), np.nan)
ypos = np.full((len(time_axis), Ntrials), np.nan)
...
vel_mag = np.sqrt(xvel**2 + yvel**2)
...
tv_disc[tv >= tongue_thresh] = 1
pv_disc[pv >= paw_thresh] = 1
me_disc[me >= me_thresh] = 1
```

iii. The agent did not explicitly frame this as unnecessary work. The notes instead present it as part of faithfully reproducing the source processing while adapting to the decoder output format.
