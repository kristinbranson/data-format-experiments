# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes a `SESSION_META` list of 44 sessions, each with animal, date, probe, task, and source directory, then loads one `data_structure_<session>.mat` per session from `data/<dir>/...`. It auto-detects MATLAB v7.3 vs v5 and reads behavior, spikes, trajectory, and SpikeGLX metadata into Python dicts. Motion energy is loaded separately from `motionEnergy_<session>.mat`.

ii.
```python
SESSION_META = [
    {'anm': 'EKH1', 'date': '2021-08-07', 'probe': 2, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    ...
]
...
data_path = os.path.join('data', data_dir, f'data_structure_{session_id}.mat')
session_data = load_session(data_path, probe)
...
me_path = os.path.join('data', data_dir, f'motionEnergy_{session_id}.mat')
me = load_motion_energy(me_path) if os.path.exists(me_path) else None
```

iii. In `CONVERSION_NOTES.md`, the agent says the session metadata are "Based on the loading scripts" and emphasizes handling both MATLAB formats and multiple motion-energy layouts. The trajectory shows it intentionally copied session/probe choices from the MATLAB loading scripts rather than globbing files.

## 1-b. How are the data split into subjects?

i. Sessions are assigned to subjects by the `anm` field in `SESSION_META`, and that value is copied into each session result as `subject`. `subjects` is then the order of first appearance across processed sessions, with `subject_idx` pointing into that list.

ii.
```python
anm = session_meta['anm']
...
result = {
    ...
    'subject': anm,
    'session_id': session_id,
}
...
if subj not in all_subjects:
    all_subjects.append(subj)
subject_idx.append(all_subjects.index(subj))
```

iii. The notes repeatedly summarize the dataset by animal names and treat `anm` as the mouse identifier, e.g. "10 names in data" and "14 unique animals."

## 1-c. How are the data split into sessions?

i. Each row of `SESSION_META` is treated as one session, identified by `<anm>_<date>`, and one output element is produced per successfully processed row. The fixed-delay and randomized-delay sessions are both handled by the same `process_session` function and differ only by the `dir` and `task` metadata.

ii.
```python
session_id = f"{anm}_{date}"
...
for sess_meta in sessions_to_process:
    result = process_session(sess_meta, PARAMS, show_processing=args.show_processing)
    if result is None:
        continue
    all_neural.append(result['neural'])
```

iii. The notes say "Include both DR and RandDelay sessions" and describe the dataset as "44 (25 DR + 19 RandDelay)" sessions.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the per-session Bpod trial count `ntrials`, and most trial-level arrays are flattened to length `ntrials`. Trial identity is therefore array index `0..ntrials-1`, while spike data use stored 1-based `trial` labels and are matched back to each trial during binning.

ii.
```python
ntrials = int(bp['Ntrials'][0, 0])
data['hit'] = bp['hit'][0, :].astype(bool)
...
for trial_idx in range(ntrials):
    trial_num = trial_idx + 1  # 1-indexed
    spike_mask = trial_nums == trial_num
```

iii. The notes describe `obj.bp` as the per-trial behavioral table and present trials/session statistics taken directly from these stored trial counts rather than reconstructed boundaries.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps only trials that are not early, not `no`/ignore, and not stimulation trials, and then further removes any trial whose neural matrix is all zeros after processing. This means ignore trials are dropped before outputs are built, and end-of-recording trials are handled by post hoc zero-neural filtering rather than by computing a recording cutoff first.

ii.
```python
valid_trial_mask = ~session_data['early'] & ~session_data['no'] & ~session_data['stim_enable']
valid_trial_mask = valid_trial_mask & (session_data['hit'] | session_data['miss'])
...
for ni, ii, oi in zip(neural_trials, input_trials, output_trials):
    if np.all(ni == 0):
        n_removed += 1
        continue
```

iii. The notes explicitly state: "Trial filtering: Exclude early, no-response, stim trials; keep hit+miss." Later notes and the trajectory justify the all-zero removal as fixing trials where recording had already stopped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from cluster-level spike times `trialtm`, trial labels `trial`, cluster quality labels `quality`, and per-trial `goCue` times for alignment. The loader does not use waveform or absolute session spike times for the final neural array.

ii.
```python
clu_info['quality'] = ...
clu_info['trialtm'] = ...
clu_info['trial'] = ...
...
goCue = session_data['goCue']
aligned_times = trialtm[spike_mask] - goCue[trial_idx]
```

iii. The notes map "clu.trialtm - goCue" to `neural` and identify the reference operations as `findClusters`, `alignSpikes`, `getSeq`, and `removeLowFRClusters`.

## 2-b. How is the `neural` data processed?

i. For each kept cluster and each trial, spikes are aligned to go cue, histogrammed into 5 ms bins from -2.5 to 2.5 s, converted to firing rate in Hz, and smoothed with a custom causal Gaussian kernel of length 15. The resulting array is stored as `(time, neurons, trials)` during processing and transposed to `(neurons, time)` per trial at output.

ii.
```python
edges = np.arange(tmin, tmax + dt, dt)
...
aligned_times = trialtm[spike_mask] - goCue[trial_idx]
counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts.astype(np.float32) / dt
fr_smooth = smooth_causal(fr, params['smooth_window'], bctype='none')
trialdat[:, neuron_idx, trial_idx] = fr_smooth
```

iii. The notes say this "matches mySmooth.m" and repeatedly cite the reference parameters `dt=1/200`, `smooth=15`, and "causal gaussian kernel."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cluster quality labels are lower-cased and stripped, then clusters labeled `garbage`, `gabrga`, `noisy`, or `real?` are removed. After binning/smoothing, units with mean firing rate `<= 0.5 Hz` across all trials and all bins are removed. Sessions are skipped if fewer than 10 quality-passing units remain before or after the firing-rate cut.

ii.
```python
PARAMS = {
    ...
    'low_fr': 0.5,
    'min_units': 10,
    'excluded_qualities': {'garbage', 'gabrga', 'noisy', 'real?'},
}
...
q = clu['quality'].lower().strip().replace('\x00', '')
if q in excluded:
    continue
...
mean_fr = np.mean(np.mean(trialdat, axis=2), axis=0)
keep_mask = mean_fr > params['low_fr']
```

iii. The notes explicitly resolve the paper/code discrepancy by choosing "Use 0.5 Hz (code default)" and say this "matches findClusters.m" while keeping empty quality strings.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting the trial's `goCue` time before binning. No extra offset or interpolation is applied to the neural stream.

ii.
```python
goCue = session_data['goCue']
...
aligned_times = trialtm[spike_mask] - goCue[trial_idx]
```

iii. The notes identify `alignEvent='goCue'` as a key reference parameter and state "Spike alignment: Matches alignSpikes.m (trialtm - goCue)."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent uses 5 ms bins (`dt = 1/200`) over a fixed window from -2.5 to 2.5 s, giving 1000 time points per trial. There is no second-stage temporal rebinning after this.

ii.
```python
'dt': 1.0 / 200.0,  # 5ms bins
...
edges = np.arange(tmin, tmax + dt, dt)
time_axis = edges[:-1] + dt / 2
```

iii. The notes repeatedly call out "Time bin: 5 ms" and "Time axis is [-2.5, 2.5] with 1000 bins at 5ms."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a stored raw variable; it is constructed from the processing parameters `tmin`, `tmax`, and `dt`, with go cue serving as the reference event. The only raw dependency is that neural and video streams are aligned relative to `goCue`.

ii.
```python
tmin, tmax, dt = params['tmin'], params['tmax'], params['dt']
edges = np.arange(tmin, tmax + dt, dt)
time_axis = edges[:-1] + dt / 2
```

iii. The notes describe the input mapping as "time_axis -> input[0] | Time from goCue in seconds."

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The agent computes the input as bin centers of the global neural time grid, then copies that same `1 x T` vector into every trial.

ii.
```python
time_axis = edges[:-1] + dt / 2
...
time_input = time_axis.reshape(1, -1).astype(np.float32)
input_trials.append(time_input)
```

iii. The notes do not claim additional processing beyond constructing the time axis from the chosen alignment window and bin size.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is exactly the same time grid used to bin neural activity, so each input sample corresponds to the center of one neural bin.

ii.
```python
counts, _ = np.histogram(aligned_times, bins=edges)
...
time_axis = edges[:-1] + dt / 2
time_input = time_axis.reshape(1, -1).astype(np.float32)
```

iii. The notes say the input time axis is "[-2.5, 2.5] with 1000 bins at 5ms," the same axis used for neural data.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. In the implemented code, lick direction is taken directly from the raw `R` trial flag after trial filtering. `hit` and `miss` are used only to choose which trials remain, not to convert instructed side into actual lick direction.

ii.
```python
valid_trial_mask = valid_trial_mask & (session_data['hit'] | session_data['miss'])
...
lick_direction = session_data['R'][valid_trial_indices].astype(int)
```

iii. The notes' variable-mapping table says "`bp.R` -> output[0] (lick_direction) | L=0, R=1 | per-trial," which matches the implementation.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. There is no reconstruction of actual lick side from hit/miss. The code simply casts `R` to integer, interprets `0` as left and `1` as right, and broadcasts that per-trial label across all time bins.

ii.
```python
lick_direction = session_data['R'][valid_trial_indices].astype(int)
...
output[0, :] = lick_direction[i]
```

iii. The notes justify this through the same direct mapping from `bp.R`, not by analyzing lick-event times or hit/miss combinations.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the raw `autowater` per-trial field.

ii.
```python
data['autowater'] = bp['autowater'][0, :]
...
context = (session_data['autowater'][valid_trial_indices] == 0).astype(int)
```

iii. The notes' mapping table says "`bp.autowater` -> output[1] (context) | WC=0, DR=1."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The agent relabels `autowater == 0` as delayed-response (`1`) and nonzero `autowater` as water-cued (`0`), then broadcasts the result across time bins.

ii.
```python
context = (session_data['autowater'][valid_trial_indices] == 0).astype(int)
...
output[1, :] = context[i]
```

iii. The notes explicitly describe this as a direct per-trial context relabeling.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `hit` flag after the agent has already filtered out `no`/ignore trials and kept only hit or miss trials.

ii.
```python
valid_trial_mask = ~session_data['early'] & ~session_data['no'] & ~session_data['stim_enable']
valid_trial_mask = valid_trial_mask & (session_data['hit'] | session_data['miss'])
...
outcome = session_data['hit'][valid_trial_indices].astype(int)
```

iii. The notes map "`bp.hit` -> output[2] (outcome) | incorrect=0, correct=1 | per-trial" and separately state that no-response trials are excluded.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The implemented output is binary rather than three-class: hit becomes `1` (`correct`), miss becomes `0` (`incorrect`), and ignore trials never appear because they were filtered out earlier. The value is then broadcast across all time bins for that trial.

ii.
```python
outcome = session_data['hit'][valid_trial_indices].astype(int)
...
output[2, :] = outcome[i]
```

iii. The notes justify this with the trial-filtering choice "keep hit+miss" and the direct `bp.hit` mapping.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The implemented tongue-velocity signal comes from the side-camera trajectory only: `traj[0]['ts']`, `traj[0]['frameTimes']`, the feature named `tongue`, plus `goCue` and the session video offset. The bottom-camera `top_tongue` feature is not used in the final computation.

ii.
```python
cam0 = session_data['traj'][0]  # side cam
...
for fi, fn in enumerate(feat_names):
    if fn.lower() == 'tongue':
        tongue_idx = fi
...
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
```

iii. The notes say "Tongue velocity: Fill NaN positions with baseline, set NaN velocity to 0" and the code comments say it is "Following reference code," but the implemented variable choice is just the side-view tongue feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, side-camera tongue `x` and `y` are linearly interpolated onto the neural `time_axis`. Missing positions are filled with the session-wide mean `x` and `y`; velocity is then the Euclidean norm of `np.gradient` on those filled traces. Bins that were NaN in the original interpolated tongue position are set to zero velocity. There is no explicit likelihood threshold, no per-run smoothing, and no combination with the bottom camera.

ii.
```python
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_ft, y, kind='linear', bounds_error=False, fill_value=np.nan)
all_x[:, trial_idx] = fx(time_axis)
all_y[:, trial_idx] = fy(time_axis)
...
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

iii. The notes explicitly justify the baseline-filling choice by citing "setTongueBaselinePosition" and say missing tongue velocity should be set to zero.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Tongue velocity is thresholded by the session-wide 50th percentile of all non-NaN tongue values from valid trials. The result is a two-class code, with NaNs reassigned to `0`; the intended third `not visible` class is not represented.

ii.
```python
def discretize_velocity(vel_data):
    valid_vals = vel_data[~np.isnan(vel_data)]
    threshold = np.percentile(valid_vals, 50)
    ...
    discretized[np.isnan(vel_data)] = 0  # default for NaN
    return discretized, threshold

tongue_disc, tongue_thresh = discretize_velocity(valid_tongue_vel)
```

iii. The notes say movement variables should be thresholded at the 50th percentile, but they do not mention preserving a separate not-visible category; the trajectory and final outputs show only binary movement classes.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The agent computes a session-wide video offset from `sglx_bitstart`, `sglx_fs`, and `bitStart`, subtracts that offset and the trial `goCue` from frame times, and then interpolates tongue position onto the neural time grid before differentiating it.

ii.
```python
vidshift = find_video_offset(session_data)
...
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
all_x[:, trial_idx] = fx(time_axis)
```

iii. The notes claim "Video offset computed matching findVideoOffset.m," and the code comments describe the output as aligned to the neural time axis via interpolation.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the second camera (`traj[1]`), preferring the feature whose name contains `top_paw`, and otherwise falling back to the first feature containing `paw`. It also uses that camera's `frameTimes`, trial `goCue`, and the same session video offset.

ii.
```python
cam1 = session_data['traj'][1]  # top cam
...
if 'top_paw' in fn.lower():
    paw_idx = fi
...
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
```

iii. The notes explicitly say "Use top_paw from top camera."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw `x` and `y` are linearly interpolated onto the neural time grid, with missing samples nearest-filled. Velocity is the Euclidean norm of the gradients of those interpolated traces after subtracting a baseline derivative computed as the median difference of each interpolated coordinate. The code does not apply an explicit likelihood threshold or the reference-style within-run Gaussian smoothing.

ii.
```python
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_ft, y, kind='linear', bounds_error=False, fill_value=np.nan)
x_interp = fx(time_axis)
y_interp = fy(time_axis)
...
arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
...
xvel = np.gradient(x_interp)
yvel = np.gradient(y_interp)
basederiv_x = np.nanmedian(np.diff(x_interp))
basederiv_y = np.nanmedian(np.diff(y_interp))
xvel -= basederiv_x
yvel -= basederiv_y
speed = np.sqrt(xvel**2 + yvel**2)
```

iii. The notes justify this as "subtract baseline derivative" for non-tongue features and using `top_paw` because it is the reliable view.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is thresholded with the same helper used for tongue velocity: session-wide median split over non-NaN values from valid trials, returning only low/high classes and mapping NaNs to `0`.

ii.
```python
paw_disc, paw_thresh = discretize_velocity(valid_paw_vel)
...
discretized[np.isnan(vel_data)] = 0
```

iii. The notes say movement variables use a per-session 50th percentile threshold, but they do not preserve the instructed third `not visible` category in the implementation.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The agent aligns paw data by subtracting the session video offset and trial `goCue` from the top-camera frame times, then interpolating paw position onto `time_axis`, which is the same grid used for neural data.

ii.
```python
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
...
x_interp = fx(time_axis)
y_interp = fy(time_axis)
```

iii. The notes describe video alignment as matching `findVideoOffset.m` and using go-cue-centered timing shared with neural data.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from a separate `motionEnergy_<session>.mat` file, specifically the per-trial `me['data']` traces. For alignment it additionally uses side-camera `frameTimes`, the session video offset, trial `goCue`, and in one fallback path `bitStart`.

ii.
```python
me_raw = d['me']
...
return {'data': trials, 'moveThresh': me_thresh}
...
ft = cam0['trials'][trial_idx]['frameTimes']
aligned_ft = ft - vidshift - goCue[trial_idx]
...
aligned_me_times = me_times - 0.5 - goCue[trial_idx] + session_data['bitStart'][trial_idx]
```

iii. The notes say motion energy comes from separate files, that three file layouts had to be handled, and that the video offset matches the reference.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion-energy traces are linearly interpolated onto the neural time axis using side-camera frame times when possible. If the motion-energy trace length does not match the frame-time vector, the agent assumes a 400 Hz motion-energy sampling grid and builds a fallback aligned time vector. After interpolation, NaNs are nearest-filled.

ii.
```python
f_interp = interp1d(aligned_ft, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, trial_idx] = f_interp(time_axis)
...
me_times = np.arange(len(trial_me)) / 400.0
aligned_me_times = me_times - 0.5 - goCue[trial_idx] + session_data['bitStart'][trial_idx]
...
col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), col[not_nans])
```

iii. The notes emphasize supporting three motion-energy file formats; the code comments justify interpolation as aligning motion energy to the neural grid.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized with the same median-split helper as the velocity outputs: values above or equal to the 50th percentile become `1`, below become `0`, and NaNs are also set to `0`. The requested third `no video` class is not represented.

ii.
```python
me_disc, me_thresh = discretize_velocity(valid_me)
...
discretized[np.isnan(vel_data)] = 0
```

iii. The notes state that motion energy should use a 50th-percentile threshold, but the code and final metadata reduce it to binary low/high categories.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by subtracting the session video offset and the trial `goCue` from side-camera frame times, then interpolating the per-frame motion-energy values onto the same neural `time_axis`. If frame counts do not match, the agent uses the 400 Hz fallback time base described above.

ii.
```python
aligned_ft = ft - vidshift - goCue[trial_idx]
f_interp = interp1d(aligned_ft, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, trial_idx] = f_interp(time_axis)
```

iii. The notes justify this through matching `findVideoOffset.m` and using the neural go-cue-centered grid for downstream decoding.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles missing data mostly by permissive fallbacks. Missing `stim.enable` becomes all-false. Missing lick arrays become empty arrays. Missing/empty probes or sessions can cause a session skip. Missing frame-times, dropped-frame metadata, empty trajectory arrays, failed motion-energy loads, and interpolation failures are silently skipped via `try/except` or `continue`. Missing motion-energy and paw values are nearest-filled; missing tongue positions are replaced with session-mean baseline coordinates and missing tongue velocities become zeros. Trials with all-zero neural matrices are removed after output construction.

ii.
```python
if 'enable' in stim:
    ...
else:
    data['stim_enable'] = np.zeros(ntrials, dtype=bool)
...
except:
    data['lickL'].append(np.array([]))
...
except Exception as e:
    print(f"  Warning: Could not load motion energy: {e}")
    return None
...
all_x_filled[np.isnan(all_x_filled)] = mean_x
...
col[nans] = np.interp(...)
...
if np.all(ni == 0):
    n_removed += 1
    continue
```

iii. The notes explicitly justify several of these choices: "Fill NaN positions with baseline, set NaN velocity to 0," and later "Removed trials with all-zero neural data" for sessions where the recording appears to have ended early.

## 11-a. What are the most time-consuming steps of the code?

i. In this implementation, the slowest steps are session loading plus the nested per-neuron, per-trial spike binning/smoothing loop and the per-trial interpolation of movement and motion-energy signals. The agent's own runtime notes report roughly 4 to 11 seconds per session and about 248 seconds total for the full conversion.

ii.
```python
session_data = load_session(data_path, probe)
...
for neuron_idx, clu in enumerate(valid_clusters):
    ...
    for trial_idx in range(ntrials):
        ...
        counts, _ = np.histogram(aligned_times, bins=edges)
...
for trial_idx in range(min(ntrials, len(cam0['trials']))):
    ...
    fx = interp1d(...)
```

iii. `CONVERSION_NOTES.md` estimates "~8s per session" and "actual: 247s" for the full run, which is consistent with these repeated per-session loops.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the double loop over neurons and trials for neural binning, the repeated per-trial interpolation loops for tongue, paw, and motion energy, and the repeated NaN-fill loops over columns or arrays. The code does not exploit a session-wide spike histogramming strategy or a shared binning helper for the video streams.

ii.
```python
for neuron_idx, clu in enumerate(valid_clusters):
    ...
    for trial_idx in range(ntrials):
        ...
```

```python
for trial_idx in range(ntrials):
    col = me_aligned[:, trial_idx]
    ...
```

iii. The trajectory shows the agent focused on correctness and decoder performance rather than optimization; the notes only give timing estimates and do not mention vectorization.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats the same pattern of per-trial interpolation and NaN handling separately for tongue, paw, and motion energy. It also reconstructs the `time_input` array independently for every trial, even though it is identical within and across sessions.

ii.
```python
fx = interp1d(...)
fy = interp1d(...)
...
for arr in [x_interp, y_interp]:
    nans = np.isnan(arr)
    ...
```

```python
time_input = time_axis.reshape(1, -1).astype(np.float32)
input_trials.append(time_input)
```

iii. The notes do not describe these repetitions as intentional; they arise from separate code paths for each output stream.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script reads several raw fields that never enter the final dataset, including `L`, `lickL`, `lickR`, `sample`, and `delay`, and it loads complete trajectory structures even though the final outputs only use a small subset of features. It also reads `moveThresh` from motion-energy files but never uses it in discretization.

ii.
```python
data['L'] = bp['L'][0, :].astype(bool)
...
data['lickL'] = []
data['lickR'] = []
...
data['sample'] = ev['sample'][0, :]
data['delay'] = ev['delay'][0, :]
...
return {'data': trials, 'moveThresh': me_thresh}
```

iii. The notes emphasize broad format coverage and reference-code exploration, which helps explain why the loader materializes more fields than the final output requires.
