# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script hard-codes a `SESSION_META` list of 44 ephys sessions, then loads one `data_structure_<animal>_<date>.mat` file per session from either `Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`. For each loaded session it reads behavioral data, cluster data, trajectory data, and SpikeGLX timing metadata; motion energy is loaded separately from `motionEnergy_<animal>_<date>.mat`.

ii.
```python
# convert_data.py lines 26-74
SESSION_META = [
    {'anm': 'EKH1', 'date': '2021-08-07', 'probe': 2, 'dir': 'Ephys_Behavior', 'task': 'DR'},
    ...
    {'anm': 'JEB24', 'date': '2023-11-03', 'probe': 1, 'dir': 'RandomizedDelay_Ephys_Behavior', 'task': 'RandDelay'},
]

# convert_data.py lines 510-518, 620-621
data_path = os.path.join('data', data_dir, f'data_structure_{session_id}.mat')
session_data = load_session(data_path, probe)

me_path = os.path.join('data', data_dir, f'motionEnergy_{session_id}.mat')
me = load_motion_energy(me_path) if os.path.exists(me_path) else None
```

iii. The code comment says the session metadata is “Based on the loading scripts in `code/DataLoadingScripts/Recording and video/`”. The notes also say the agent chose to include “both DR and RandDelay sessions”.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `anm` field in each session’s metadata. The script builds a unique ordered `subjects` list and a `subject_idx` array that maps each processed session to its subject.

ii.
```python
# convert_data.py lines 501-505, 1073-1076, 1088-1089
anm = session_meta['anm']
...
subj = result['subject']
if subj not in all_subjects:
    all_subjects.append(subj)
subject_idx.append(all_subjects.index(subj))
...
'subjects': all_subjects,
'subject_idx': np.array(subject_idx),
```

iii. There is no deeper justification beyond using the animal name from the session metadata; the notes summarize the output as 14 unique animals.

## 1-c. How are the data split into sessions?

i. Each entry of `SESSION_META` is treated as one session. `process_session` returns one session-level result, and the top-level lists `neural`, `input`, and `output` each append one item per processed session.

ii.
```python
# convert_data.py lines 1044-1069, 1084-1087
sessions_to_process = SESSION_META
...
for sess_meta in sessions_to_process:
    result = process_session(sess_meta, PARAMS, show_processing=args.show_processing)
    if result is None:
        continue
    all_neural.append(result['neural'])
    all_input.append(result['input'])
    all_output.append(result['output'])
...
'neural': all_neural,
'input': all_input,
'output': all_output,
```

iii. The trajectory and notes show the agent manually compiled the 44-session list from the MATLAB loading scripts and treated each entry as one session.

## 1-d. How are the data split into trials?

i. Within a session, the script first allocates trial-major arrays using `bp.Ntrials`, computes neural and behavioral variables for all trials, then keeps only `valid_trial_indices`. Each kept trial becomes one `(neurons, time)` neural matrix, one `(1, time)` input array, and one `(6, time)` output array.

ii.
```python
# convert_data.py lines 524, 552-555, 602, 873-893
ntrials = session_data['ntrials']
trialdat = np.zeros((n_timepts, n_neurons, ntrials), dtype=np.float32)
...
valid_trial_indices = np.where(valid_trial_mask)[0]
...
for i, trial_idx in enumerate(valid_trial_indices):
    neural = trialdat[:, :, trial_idx].T.copy()
    neural_trials.append(neural.astype(np.float32))
    time_input = time_axis.reshape(1, -1).astype(np.float32)
    input_trials.append(time_input)
    output = np.zeros((6, n_timepts), dtype=np.int64)
    ...
    output_trials.append(output)
```

iii. The notes describe this as excluding invalid trials after building per-trial data and then packaging the remaining trials into the decoder format.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by excluding `early` trials, `no`-response trials, and stimulation trials, and by requiring each kept trial to be either a hit or a miss. After packaging, trials whose neural matrix is entirely zero are also dropped.

ii.
```python
# convert_data.py lines 596-607, 898-914
valid_trial_mask = ~session_data['early'] & ~session_data['no'] & ~session_data['stim_enable']
valid_trial_mask = valid_trial_mask & (session_data['hit'] | session_data['miss'])
valid_trial_indices = np.where(valid_trial_mask)[0]
...
for ni, ii, oi in zip(neural_trials, input_trials, output_trials):
    if np.all(ni == 0):
        n_removed += 1
        continue
```

iii. The notes say “Trial filtering: Exclude early, no-response, stim trials; keep hit+miss”, and later explain the all-zero-neural trial removal as cleanup for end-of-session recording dropouts.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times and trial IDs in `clu.trialtm` and `clu.trial`, with alignment to `bp.ev.goCue`.

ii.
```python
# convert_data.py lines 217-227, 357-361, 550, 557-567
clu_info['trialtm'] = f[ref][()].flatten()
clu_info['trial'] = f[ref][()].flatten().astype(int)
...
clu_info['trialtm'] = c['trialtm'].flatten().astype(float)
clu_info['trial'] = c['trial'].flatten().astype(int)
...
goCue = session_data['goCue']
trialtm = clu['trialtm']
trial_nums = clu['trial']
aligned_times = trialtm[spike_mask] - goCue[trial_idx]
```

iii. The notes explicitly map neural data to “`clu.trialtm - goCue`”.

## 2-b. How is the `neural` data processed?

i. For each kept cluster and each trial, spike times are aligned to go cue, histogrammed into 5 ms bins over `[-2.5, 2.5]`, converted to firing rate by dividing by `dt`, and smoothed with a causal Gaussian kernel of width 15.

ii.
```python
# convert_data.py lines 543-575
edges = np.arange(tmin, tmax + dt, dt)
time_axis = edges[:-1] + dt / 2
...
aligned_times = trialtm[spike_mask] - goCue[trial_idx]
counts, _ = np.histogram(aligned_times, bins=edges)
fr = counts.astype(np.float32) / dt
fr_smooth = smooth_causal(fr, params['smooth_window'], bctype='none')
trialdat[:, neuron_idx, trial_idx] = fr_smooth
```

iii. The notes and code comments say this is meant to match `alignSpikes.m`, `getSeq.m`, and `mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script removes clusters whose quality label is one of `garbage`, `gabrga`, `noisy`, or `real?`, then removes clusters with mean firing rate `<= 0.5 Hz`, and finally skips sessions with fewer than 10 remaining units.

ii.
```python
# convert_data.py lines 526-541, 577-591
excluded = params['excluded_qualities']
valid_clusters = []
for i, clu in enumerate(session_data['clusters']):
    q = clu['quality'].lower().strip().replace('\x00', '')
    if q in excluded:
        continue
    valid_clusters.append(clu)
...
mean_fr = np.mean(np.mean(trialdat, axis=2), axis=0)
keep_mask = mean_fr > params['low_fr']
...
if n_kept < params['min_units']:
    print(f"    WARNING: Only {n_kept} units after FR filter, skipping session")
    return None
```

iii. The notes justify the quality filter as matching `findClusters.m` and explicitly choose the code’s `0.5 Hz` low-FR threshold rather than the paper text’s `1 Hz`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial’s spikes are aligned by subtracting that trial’s `goCue` time from the within-trial spike times before binning. The output window is from `-2.5 s` to `+2.5 s` relative to go cue.

ii.
```python
# convert_data.py lines 76-85, 543-550, 566-570
PARAMS = {
    'tmin': -2.5,
    'tmax': 2.5,
    'align_event': 'goCue',
}
...
goCue = session_data['goCue']
aligned_times = trialtm[spike_mask] - goCue[trial_idx]
counts, _ = np.histogram(aligned_times, bins=edges)
```

iii. The notes repeatedly state “Align spikes to goCue” and “Time window: `[-2.5, 2.5]s from goCue`”.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data is represented at 5 ms resolution (`dt = 1/200`). No later rebinning is applied in the conversion script.

ii.
```python
# convert_data.py lines 77-80, 543-547
PARAMS = {
    'dt': 1.0 / 200.0,  # 5ms bins
}
...
edges = np.arange(tmin, tmax + dt, dt)
time_axis = edges[:-1] + dt / 2
```

iii. The notes justify this directly from `getDefaultParams.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from a raw sampled signal. It is synthesized from the alignment choice (`goCue`) and the fixed binning parameters `tmin`, `tmax`, and `dt`.

ii.
```python
# convert_data.py lines 543-547, 878-880
edges = np.arange(tmin, tmax + dt, dt)
time_axis = edges[:-1] + dt / 2
...
time_input = time_axis.reshape(1, -1).astype(np.float32)
```

iii. The notes map the decoder input to “`time_axis`” rather than to a raw data field.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script creates a fixed vector of bin centers from `-2.5` to `+2.5` seconds in 5 ms steps and reuses the same vector for every valid trial in the session.

ii.
```python
# convert_data.py lines 543-547, 873-881
edges = np.arange(tmin, tmax + dt, dt)
time_axis = edges[:-1] + dt / 2
...
for i, trial_idx in enumerate(valid_trial_indices):
    ...
    time_input = time_axis.reshape(1, -1).astype(np.float32)
    input_trials.append(time_input)
```

iii. The justification is implicit: the task asked for time from go-cue onset, and the notes say the input should just be the aligned time axis.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the exact same `time_axis` that is used to bin and smooth the neural data, so it is sample-aligned by construction.

ii.
```python
# convert_data.py lines 543-547, 573-575, 878-880
time_axis = edges[:-1] + dt / 2
...
fr_smooth = smooth_causal(fr, params['smooth_window'], bctype='none')
trialdat[:, neuron_idx, trial_idx] = fr_smooth
...
time_input = time_axis.reshape(1, -1).astype(np.float32)
```

iii. The notes describe both neural and input streams as sharing the `[-2.5, 2.5]` go-cue-centered axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived from the per-trial boolean `bp.R` field, with `bp.L` only implicit in the complement.

ii.
```python
# convert_data.py lines 159-160, 305-306, 609-612
data['R'] = bp['R'][0, :].astype(bool)
data['L'] = bp['L'][0, :].astype(bool)
...
lick_direction = session_data['R'][valid_trial_indices].astype(int)
```

iii. The notes justify this as “`bp.R` -> output[0] (lick_direction) | `L=0, R=1`”.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The script casts the selected `R` trials to integers and then broadcasts the per-trial label across all time bins of that trial’s output matrix.

ii.
```python
# convert_data.py lines 610-612, 886-888
lick_direction = session_data['R'][valid_trial_indices].astype(int)
...
output = np.zeros((6, n_timepts), dtype=np.int64)
output[0, :] = lick_direction[i]
```

iii. No additional justification appears beyond the notes’ simple left/right mapping.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the `bp.autowater` field.

ii.
```python
# convert_data.py lines 161, 307, 613-615
data['autowater'] = bp['autowater'][0, :]
...
data['autowater'] = bp['autowater'].flatten()
...
context = (session_data['autowater'][valid_trial_indices] == 0).astype(int)
```

iii. The notes explicitly state “`bp.autowater` -> context | `WC=0, DR=1`”.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The script maps `autowater == 0` to `1` (DR) and all other kept values to `0` (WC), then broadcasts that per-trial label across time.

ii.
```python
# convert_data.py lines 613-615, 887-889
context = (session_data['autowater'][valid_trial_indices] == 0).astype(int)
...
output[1, :] = context[i]
```

iii. The notes say the agent chose this mapping because `autowater` distinguishes WC from DR blocks in the reference code.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `bp.hit` field after no-response trials have already been filtered out.

ii.
```python
# convert_data.py lines 155-157, 301-303, 616-617
data['hit'] = bp['hit'][0, :].astype(bool)
data['miss'] = bp['miss'][0, :].astype(bool)
data['no'] = bp['no'][0, :].astype(bool)
...
outcome = session_data['hit'][valid_trial_indices].astype(int)
```

iii. The notes say “`bp.hit` -> outcome | incorrect=0, correct=1”.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The script casts `hit` to integer, which makes hit trials `1` and miss trials `0`, and then broadcasts the per-trial label over time.

ii.
```python
# convert_data.py lines 616-617, 888-890
outcome = session_data['hit'][valid_trial_indices].astype(int)
...
output[2, :] = outcome[i]
```

iii. This matches the notes’ binary correct/incorrect mapping; there is no additional processing.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-camera DeepLabCut trajectory stream: `traj[0]`, the feature named exactly `tongue`, and its x/y coordinates from `trial_info['ts']`.

ii.
```python
# convert_data.py lines 679-688, 702-713
cam0 = session_data['traj'][0]  # side cam
feat_names = cam0['featNames']
...
if fn.lower() == 'tongue':
    tongue_idx = fi
...
if ts.shape[0] == len(feat_names):
    x = ts[tongue_idx, 0, :].copy()
    y = ts[tongue_idx, 1, :].copy()
elif ts.shape[2] == len(feat_names):
    x = ts[:, 0, tongue_idx].copy()
    y = ts[:, 1, tongue_idx].copy()
```

iii. The notes justify this at a high level as “DLC tongue velocity”, and the trajectory shows the agent intentionally picked the single `tongue` feature from the side camera.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The script interpolates tongue x/y positions to the neural time axis, fills missing positions with a session-wide mean baseline, computes `np.gradient` of the filled positions, zeros the gradient wherever the original position was NaN, and then takes Euclidean speed.

ii.
```python
# convert_data.py lines 718-757
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_ft, y, kind='linear', bounds_error=False, fill_value=np.nan)
all_x[:, trial_idx] = fx(time_axis)
all_y[:, trial_idx] = fy(time_axis)
...
mean_x = np.nanmean(all_x)
mean_y = np.nanmean(all_y)
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

iii. The notes justify this as “fill NaN positions with baseline, set NaN velocity to 0”; the trajectory shows the agent struggled with tongue sparsity and intentionally added the baseline-fill step.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The script computes a per-session median over all non-NaN tongue-speed samples from valid trials. Normally it thresholds with `>= median`, but if the median equals the minimum value it switches to strict `>` to avoid producing an all-ones label.

ii.
```python
# convert_data.py lines 846-863
def discretize_velocity(vel_data):
    valid_vals = vel_data[~np.isnan(vel_data)]
    ...
    threshold = np.percentile(valid_vals, 50)
    if threshold <= np.min(valid_vals) + 1e-10:
        discretized = (vel_data > threshold).astype(int)
    else:
        discretized = (vel_data >= threshold).astype(int)
    discretized[np.isnan(vel_data)] = 0
    return discretized, threshold

tongue_disc, tongue_thresh = discretize_velocity(valid_tongue_vel)
```

iii. The trajectory explicitly says the agent changed the `>=` rule to `>` in the zero-median case because the literal median rule made almost all tongue samples class `1`.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue positions are aligned by subtracting both the estimated video offset and the trial’s `goCue` time from the video frame times, then interpolating onto the neural `time_axis`.

ii.
```python
# convert_data.py lines 484-496, 718-724
def find_video_offset(session_data):
    ...
    vidshift = vidFileOffset - bitStart
    return vidshift
...
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_ft, y, kind='linear', bounds_error=False, fill_value=np.nan)
all_x[:, trial_idx] = fx(time_axis)
all_y[:, trial_idx] = fy(time_axis)
```

iii. The notes say the video offset computation is meant to match `findVideoOffset.m`, and the alignment is meant to match the reference `findPosition.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the second camera’s DeepLabCut trajectory stream, preferentially the feature whose name contains `top_paw`, otherwise the first feature containing `paw`.

ii.
```python
# convert_data.py lines 761-775, 789-795
cam1 = session_data['traj'][1]  # top cam
feat_names = cam1['featNames']
...
if 'top_paw' in fn.lower():
    paw_idx = fi
    break
...
elif ts.shape[2] == len(feat_names):
    x = ts[:, 0, paw_idx]
    y = ts[:, 1, paw_idx]
```

iii. The notes justify this choice as “Use `top_paw` from top camera”.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The script interpolates paw x/y positions to the neural time axis, fills missing positions by nearest interpolation, computes `np.gradient`, subtracts a per-axis median derivative as baseline, fills remaining NaNs by nearest interpolation, and computes Euclidean speed.

ii.
```python
# convert_data.py lines 801-834
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
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
...
speed = np.sqrt(xvel**2 + yvel**2)
```

iii. The notes summarize this as “subtract baseline derivative” for the non-tongue feature, following the reference kinematics functions.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw speed uses the same `discretize_velocity` helper as tongue speed: per-session 50th percentile over non-NaN values, with the same strict-`>` fallback when the percentile equals the minimum.

ii.
```python
# convert_data.py lines 846-864
threshold = np.percentile(valid_vals, 50)
if threshold <= np.min(valid_vals) + 1e-10:
    discretized = (vel_data > threshold).astype(int)
else:
    discretized = (vel_data >= threshold).astype(int)
...
paw_disc, paw_thresh = discretize_velocity(valid_paw_vel)
```

iii. The notes do not separately justify paw thresholding; it inherits the generic velocity-threshold logic documented in the code and trajectory.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw positions are aligned using `frameTimes - vidshift - goCue` and then interpolated directly to the neural `time_axis`.

ii.
```python
# convert_data.py lines 801-809
aligned_ft = ft[:len(x)] - vidshift - goCue[trial_idx]
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_ft, y, kind='linear', bounds_error=False, fill_value=np.nan)
x_interp = fx(time_axis)
y_interp = fy(time_axis)
```

iii. The code comments and notes present this as using the same video-to-neural alignment strategy as the reference functions.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<session>.mat` file, specifically the `me` variable or its nested/direct variants.

ii.
```python
# convert_data.py lines 428-478, 620-621
def load_motion_energy(me_filepath):
    d = sio.loadmat(me_filepath, squeeze_me=False)
    me_raw = d['me']
    ...
    return {'data': trials, 'moveThresh': me_thresh}
...
me_path = os.path.join('data', data_dir, f'motionEnergy_{session_id}.mat')
me = load_motion_energy(me_path) if os.path.exists(me_path) else None
```

iii. The notes say the agent added support for three motion-energy file layouts after finding format differences in the data.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The script loads the trialwise motion-energy traces, interpolates them to the neural time axis using side-camera frame times when possible, otherwise synthesizes a 400 Hz time base, and finally fills internal NaNs by nearest-value interpolation.

ii.
```python
# convert_data.py lines 631-674
for trial_idx in range(min(ntrials, len(me['data']))):
    trial_me = me['data'][trial_idx]
    ...
    aligned_ft = ft - vidshift - goCue[trial_idx]
    if len(trial_me) == len(ft):
        f_interp = interp1d(aligned_ft, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
        me_aligned[:, trial_idx] = f_interp(time_axis)
    else:
        me_times = np.arange(len(trial_me)) / 400.0
        aligned_me_times = me_times - 0.5 - goCue[trial_idx] + session_data['bitStart'][trial_idx]
        f_interp = interp1d(aligned_me_times, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
        me_aligned[:, trial_idx] = f_interp(time_axis)
...
col[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), col[not_nans])
```

iii. The notes justify this as matching `loadMotionEnergy.m` while also handling extra data formats discovered during debugging.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded with the same `discretize_velocity` helper used for tongue and paw speed: per-session 50th percentile over non-NaN aligned values, plus the strict-`>` fallback when the 50th percentile equals the minimum.

ii.
```python
# convert_data.py lines 846-864
threshold = np.percentile(valid_vals, 50)
if threshold <= np.min(valid_vals) + 1e-10:
    discretized = (vel_data > threshold).astype(int)
else:
    discretized = (vel_data >= threshold).astype(int)
...
me_disc, me_thresh = discretize_velocity(valid_me)
```

iii. The notes say the agent intentionally replaced the paper’s manual motion-energy threshold with the decoder task’s requested per-session median split.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by subtracting video offset and `goCue` from the video frame times and then interpolating onto the neural `time_axis`; when frame-time length does not match trace length, a fallback synthetic 400 Hz axis is used instead.

ii.
```python
# convert_data.py lines 637-660
ft = cam0['trials'][trial_idx]['frameTimes']
aligned_ft = ft - vidshift - goCue[trial_idx]
...
f_interp = interp1d(aligned_ft, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
...
me_times = np.arange(len(trial_me)) / 400.0
aligned_me_times = me_times - 0.5 - goCue[trial_idx] + session_data['bitStart'][trial_idx]
f_interp = interp1d(aligned_me_times, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
```

iii. The notes say this was meant to match `loadMotionEnergy.m`; the trajectory shows the fallback path was added after debugging sessions with alternative ME formats.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses broad `try/except` blocks, defaults missing arrays to empty arrays or zeros, keeps unlabeled clusters, fills missing motion-energy and non-tongue kinematic samples by interpolation, fills missing tongue positions with a baseline and missing tongue velocity with zeros, falls back to `vidshift = 0`, and removes trials with all-zero neural activity.

ii.
```python
# convert_data.py lines 183-193, 435-481, 624-627, 665-674, 728-740, 751-754, 811-815, 898-906
try:
    ll = f[lickL_refs[i]][()].flatten()
    data['lickL'].append(ll)
except:
    data['lickL'].append(np.array([]))
...
except Exception as e:
    print(f"  Warning: Could not load motion energy: {e}")
    return None
...
try:
    vidshift = find_video_offset(session_data)
except:
    vidshift = 0.0
...
col[nans] = np.interp(...)
...
all_x_filled[np.isnan(all_x_filled)] = mean_x
...
xvel[nan_mask] = 0.0
yvel[nan_mask] = 0.0
...
if np.all(ni == 0):
    n_removed += 1
    continue
```

iii. The notes justify several of these explicitly: keeping empty quality strings to match `findClusters.m`, filling tongue NaNs with baseline, and removing all-zero end-of-session trials.

## 11-a. What are the most time-consuming steps of the code?

i. The most expensive work is the nested neuron-by-trial spike binning/smoothing loop, plus the trialwise interpolation loops for motion energy and video-based kinematics.

ii.
```python
# convert_data.py lines 556-575, 631-674, 695-757, 778-835
for neuron_idx, clu in enumerate(valid_clusters):
    ...
    for trial_idx in range(ntrials):
        ...
        counts, _ = np.histogram(aligned_times, bins=edges)
        fr_smooth = smooth_causal(fr, params['smooth_window'], bctype='none')
...
for trial_idx in range(min(ntrials, len(me['data']))):
    ...
for trial_idx in range(min(ntrials, len(cam0['trials']))):
    ...
for trial_idx in range(min(ntrials, len(cam1['trials']))):
    ...
```

iii. The notes’ runtime discussion (“~8s per session”) and the structure of the code both point to these nested loops as the dominant cost.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner `for trial_idx in range(ntrials)` loop inside the neuron loop could be replaced by grouping spikes by trial more efficiently, and the per-trial interpolation loops for motion energy, tongue, and paw could also be partially vectorized or batched.

ii.
```python
# convert_data.py lines 556-575, 631-674, 695-757, 778-835
for neuron_idx, clu in enumerate(valid_clusters):
    for trial_idx in range(ntrials):
        ...
for trial_idx in range(min(ntrials, len(me['data']))):
    ...
for trial_idx in range(min(ntrials, len(cam0['trials']))):
    ...
for trial_idx in range(min(ntrials, len(cam1['trials']))):
    ...
```

iii. There is no explicit justification from the agent; this follows from the code’s structure.

## 11-c. What processing does the code repeat multiple times?

i. It repeats the same interpolation-and-fill pattern separately for motion energy, tongue positions, and paw positions, and it recomputes the same `time_input` array for every trial instead of sharing one object per session.

ii.
```python
# convert_data.py lines 647-650, 721-724, 806-809, 873-881
f_interp = interp1d(aligned_ft, trial_me, kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, trial_idx] = f_interp(time_axis)
...
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_ft, y, kind='linear', bounds_error=False, fill_value=np.nan)
...
fx = interp1d(aligned_ft, x, kind='linear', bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_ft, y, kind='linear', bounds_error=False, fill_value=np.nan)
...
time_input = time_axis.reshape(1, -1).astype(np.float32)
input_trials.append(time_input)
```

iii. This is not justified in the notes; it is a byproduct of straightforward, duplicated implementation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `lickL`, `lickR`, `sample`, and `delay` but never uses them in the final dataset; it computes and stores motion-energy `moveThresh` even though the final labels are recomputed by median split; and it optionally generates large processing plots that are not part of the output dataset.

ii.
```python
# convert_data.py lines 173-193, 428-478, 926-1028
data['sample'] = ev['sample'][0, :]
data['delay'] = ev['delay'][0, :]
data['lickL'] = []
data['lickR'] = []
...
return {'data': trials, 'moveThresh': me_thresh}
...
if show_processing:
    ...
    plt.savefig(f'processing_{session_id}.png', dpi=100)
```

iii. There is no explicit justification except convenience during development; the notes mention the optional processing plots as debugging/validation aids.
