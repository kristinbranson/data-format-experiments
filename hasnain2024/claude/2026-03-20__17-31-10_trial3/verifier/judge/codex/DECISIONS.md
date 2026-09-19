# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-coded a 44-session manifest, including the folder, animal, date, and probe numbers for each session. It loads each `data_structure_<animal>_<date>.mat` file with either `h5py` or `scipy.io.loadmat`, and it loads motion energy separately from `motionEnergy_<animal>_<date>.mat`.

ii. <Code snippets>
```python
EPHYS_SESSIONS = [
    ('data/Ephys_Behavior', 'JEB6', '2021-04-18', [2]),
    ...
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-11-03', [1]),
]
```
```python
def load_mat_file(filepath):
    try:
        f = h5py.File(filepath, 'r')
        return f, 'h5'
    except Exception:
        mat = scipy.io.loadmat(filepath, squeeze_me=False)
        return mat, 'v5'
```
```python
filepath = os.path.join(dirpath, f"data_structure_{session_id}.mat")
data, fmt = load_mat_file(filepath)
...
me_data, me_thresh = load_motion_energy(dirpath, animal, date)
```

iii. The notes say the session list came from the paper's loader scripts, and the AI explicitly chose to include only the two ephys directories and not the behavior-only inhibition datasets. It justified dual `.mat` readers because the shared files mix HDF5 v7.3 and MATLAB v5 formats.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the `animal` field in each hard-coded session tuple. At assembly, the AI takes the sorted unique animal names as `subjects` and maps each session to `subject_idx`.

ii. <Code snippets>
```python
for sess_idx, (dirpath, animal, date, probes) in enumerate(sessions):
    ...
    all_animals.append(animal)
```
```python
unique_subjects = sorted(set(all_animals))
subject_idx = np.array([unique_subjects.index(a) for a in all_animals])
```

iii. The notes justify this by treating the manifest as authoritative for subject identity and by reporting 14 unique animals across the 44 retained sessions.

## 1-c. How are the data split into sessions?

i. One tuple in `EPHYS_SESSIONS` is one session, and one session becomes one element in `neural`, `input`, and `output`. The code iterates session-by-session and skips a session only if it fails basic checks such as too few valid trials or too few units.

ii. <Code snippets>
```python
for sess_idx, (dirpath, animal, date, probes) in enumerate(sessions):
    result = process_session(
        dirpath, animal, date, probes, time_axis, edges,
        show_processing=args.show_processing, session_idx=sess_idx
    )

    if result is None:
        continue
```
```python
data = {
    'neural': all_neural,
    'input': all_input,
    'output': all_output,
    ...
}
```

iii. The notes say the AI followed the authors' loader scripts and treated the fixed-delay and randomized-delay folders as one combined ephys dataset for decoder use.

## 1-d. How are the data split into trials?

i. Trials are indexed from `bp.Ntrials`, and the code treats trial `j` as MATLAB trial number `j + 1`. Behavioral arrays are flattened once, and valid trials are the indices where the trial-level mask passes. Neural and video data are then pulled per trial using those indices.

ii. <Code snippets>
```python
def get_ntrials(data, fmt):
    if fmt == 'h5':
        return int(data['obj']['bp']['Ntrials'][0, 0])
    else:
        return int(data['obj']['bp'][0, 0]['Ntrials'][0, 0])
```
```python
valid_trial_indices = np.where(valid_trials)[0]
...
for j in range(ntrials):
    trial_num = j + 1  # 1-indexed
    spike_mask = trial == trial_num
```

iii. The notes describe the behavioral fields as one entry per trial and assume trial indexing can be taken directly from `Ntrials`. There is no explicit justification for not truncating overlong fields to `Ntrials`.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops early-lick trials and photostimulation trials, then also drops trials whose go-cue time is missing or non-positive. After loading spikes, it additionally drops valid behavioral trials beyond the last trial with any spike data.

ii. <Code snippets>
```python
valid_trials = ~stim_enable & ~early
valid_trials &= ~np.isnan(align_times) & (align_times > 0)
valid_trial_indices = np.where(valid_trials)[0]
```
```python
max_spike_trial = max(
    int(np.max(c['trial'])) for c in all_clusters if len(c['trial']) > 0
)
...
valid_trial_indices = valid_trial_indices[valid_trial_indices < max_spike_trial]
```

iii. The notes justify removing `stim.enable` and `early` from the reference pipeline, and later justify the recording-end cutoff after seeing sessions with zero neural data in the last behavioral trials. The extra go-cue validity filter is justified only implicitly as a safety check.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural arrays come from `obj.clu` cluster fields `trialtm`, `trial`, and `quality`, plus `bp.ev.goCue` for alignment. Probe selection comes from the session manifest.

ii. <Code snippets>
```python
clusters.append({
    'trialtm': trialtm,
    'trial': trial,
    'quality': quality,
})
```
```python
align_times = get_event_times(data, fmt, ALIGN_EVENT)
...
aligned_times = trialtm[spike_mask] - align_times[j]
```

iii. The notes explicitly map neural data to spike times per cluster, cluster quality labels, and go-cue times, following the reference code structure.

## 2-b. How is the `neural` data processed?

i. The AI bins spikes into 10 ms bins from -2.5 s to 2.5 s around the go cue, converts counts to firing rates by dividing by `DT`, and applies a custom causal Gaussian smoothing filter with window length 15 and reflect padding.

ii. <Code snippets>
```python
DT = 1.0 / 100  # 10 ms time bins
SMOOTH_N = 15
SMOOTH_BC = 'reflect'
```
```python
counts = np.histogram(aligned_times, bins=edges)[0]
rate = counts.astype(np.float64) / DT
smoothed = causal_gaussian_smooth(rate, SMOOTH_N, SMOOTH_BC)
trialdat[:, i, j] = smoothed.astype(np.float32)
```

iii. The notes say the AI chose 10 ms because `WorkingWithDataObjs.m` uses `dt = 1/100`, despite also noting that `getDefaultParams.m` uses 5 ms. It further states that smoothing should match `mySmooth.m` and uses a causal kernel for that reason.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code first removes clusters whose quality label is one of `garbage`, `gabrga`, `noisy`, or `real?`. It then removes neurons with mean firing rate `<= 1 Hz`, and finally drops any session with fewer than 10 neurons after filtering.

ii. <Code snippets>
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR_THRESHOLD = 1.0
MIN_UNITS = 10
```
```python
if quality in EXCLUDE_QUALITIES:
    continue
```
```python
mean_frs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
keep = mean_frs > LOW_FR_THRESHOLD
...
if n_neurons < MIN_UNITS:
    return None
```

iii. The notes justify the quality list from `findClusters.m`, the `>1 Hz` cutoff from the paper, and the session-level `>= 10 units` rule from the methods text.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting each trial's go-cue time from that trial's `trialtm` values before binning.

ii. <Code snippets>
```python
ALIGN_EVENT = 'goCue'
...
align_times = get_event_times(data, fmt, ALIGN_EVENT)
...
aligned_times = trialtm[spike_mask] - align_times[j]
```

iii. The notes explicitly tie this to the reference alignment step `trialtm_aligned = trialtm - event_time`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural, input, and time-varying output data use 10 ms bins over a 5 s window, for 500 bins total. No secondary rebinning is applied after the initial binning/interpolation onto this grid.

ii. <Code snippets>
```python
DT = 1.0 / 100  # 10 ms time bins
...
edges = np.arange(TMIN, TMAX + DT/2, DT)
time_axis = edges[:-1] + DT / 2
```
```python
print(f"Time axis: {time_axis[0]:.4f} to {time_axis[-1]:.4f} s, {n_timepoints} bins, dt={DT*1000:.1f} ms")
```

iii. The notes justify 10 ms by preferring `WorkingWithDataObjs.m` over the 5 ms setting found elsewhere in the reference code.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a raw data field directly. It is a synthetic time axis defined relative to the go cue using the global window constants `TMIN`, `TMAX`, and `DT`, with the go cue as the alignment event.

ii. <Code snippets>
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 100
```
```python
def compute_time_axis():
    edges = np.arange(TMIN, TMAX + DT/2, DT)
    time_axis = edges[:-1] + DT / 2
    return time_axis, edges
```

iii. The notes describe this variable as "time from goCue" and justify it as the shared decoder input axis rather than a field stored in the raw files.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes bin edges over `[-2.5, 2.5]` s and uses the bin centers as the input value. It then repeats the same `(1, n_timepoints)` array for every trial in a session.

ii. <Code snippets>
```python
edges = np.arange(TMIN, TMAX + DT/2, DT)
time_axis = edges[:-1] + DT / 2
```
```python
input_data = time_axis.astype(np.float32).reshape(1, -1)
input_trials.append(input_data)
```

iii. The notes say this was meant to mirror the neural bin centers and provide the only decoder input variable. The chosen bin size was justified by the 10 ms interpretation of the reference code.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time-from-go-cue input uses the exact same `time_axis` that neural spike counts are binned onto, so its time points correspond directly to the neural bins.

ii. <Code snippets>
```python
time_axis, edges = compute_time_axis()
trialdat = bin_and_smooth_spikes(all_clusters, ntrials, align_times, time_axis, edges)
```
```python
input_data = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes explicitly describe the time axis as the common alignment grid for all streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Although the code reads `hit`, `miss`, `R`, and `L`, the final implementation derives lick direction only from `R`, meaning it uses the instructed or trial side rather than the animal's actual lick direction.

ii. <Code snippets>
```python
hit = get_bp_field(data, fmt, 'hit').astype(bool)
miss = get_bp_field(data, fmt, 'miss').astype(bool)
R = get_bp_field(data, fmt, 'R').astype(bool)
L = get_bp_field(data, fmt, 'L').astype(bool)
```
```python
# Lick direction: R=1, L=0 (trial instruction/stimulus side)
lick_direction = R_valid.astype(np.float32)
```

iii. The notes show the AI noticed the difference between instructed side and actual lick direction, but then "kept it simple" and used `R/L` as the trial-type variable.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. No behavioral inference is performed in the final code. The code casts `R_valid` to `0/1` and repeats that scalar across all time bins of the trial, with output labels only `left` and `right`.

ii. <Code snippets>
```python
lick_direction = R_valid.astype(np.float32)
...
out[0, :] = int(lick_direction[t_idx])
```
```python
'output_values': [
    ['left', 'right'],
    ...
]
```

iii. The notes justify this as a simpler interpretation of the trial side, even though earlier notes recognized that actual lick direction would require combining trial side with hit/miss and a no-lick class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the trial-level `autowater` flag in `obj.bp`.

ii. <Code snippets>
```python
autowater = get_bp_field(data, fmt, 'autowater').astype(bool)
```
```python
behavioral_context = (~autowater_valid).astype(np.float32)
```

iii. The notes justify this directly from the methods: `autowater` marks the water-cued context and its complement marks delayed response.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code inverts `autowater`, so `WC` is encoded as `0` and `DR` as `1`. The resulting per-trial scalar is broadcast across the time bins of that trial.

ii. <Code snippets>
```python
# Behavioral context: WC=0, DR=1
behavioral_context = (~autowater_valid).astype(np.float32)
...
out[1, :] = int(behavioral_context[t_idx])
```

iii. The notes explicitly justify the coding as `WC=0`, `DR=1` to match the decoder specification.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The code reads both `hit` and `miss`, but the final implementation uses only `hit` when building `outcome`.

ii. <Code snippets>
```python
hit = get_bp_field(data, fmt, 'hit').astype(bool)
miss = get_bp_field(data, fmt, 'miss').astype(bool)
```
```python
# Outcome: correct=1, incorrect=0
outcome = hit_valid.astype(np.float32)
```

iii. The notes initially discuss misses and no-response trials, but the final implementation collapses everything non-hit into one class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is encoded as binary `correct` versus `not correct`: hit trials become `1`, while miss and ignore trials both become `0`. The code does not create a separate ignore class.

ii. <Code snippets>
```python
outcome = hit_valid.astype(np.float32)
...
out[2, :] = int(outcome[t_idx])
```
```python
'output_values': [
    ...,
    ['incorrect', 'correct'],
    ...
]
```

iii. The notes justify this as a simple decoder output, but they do not give a reference-based justification for collapsing misses and ignores together.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-camera `traj` stream, using only the feature named `tongue`, its `frameTimes`, and the `ts` position array. The code also uses `bp.ev.goCue` and a session-wide video offset from `sglx`/`bp.ev.bitStart`.

ii. <Code snippets>
```python
tongue_speed = extract_velocity_from_traj(
    data, fmt, view=0, feat_name='tongue',
    ntrials=ntrials, align_times=align_times,
    time_axis=time_axis, vidshift=vidshift
)
```
```python
ts, frame_times, feat_names = get_traj_data(data, fmt, view, trial_idx)
...
feat_idx = feat_names.index(feat_name)
```

iii. The notes show the AI considered other tracked features, even mentioning `jaw` as a possible proxy, but ultimately chose the side-camera tongue because the task name said "tongue velocity".

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the code extracts side-camera tongue positions, aligns frame times to the go cue, computes simple position gradients, converts NaN tongue velocities to zero, fills missing values with nearest neighbors, and linearly interpolates the result onto the session time axis. It does not use likelihood thresholds, bottom-camera tongue tracking, run-wise smoothing, or per-frame-time derivatives.

ii. <Code snippets>
```python
if len(frame_times) == 0 or np.all(np.isnan(frame_times)):
    frame_times = np.arange(n_frames) / 400.0
    ft_offset = 0.5
else:
    ft_offset = vidshift
...
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
```
```python
is_tongue = 'tongue' in feat_name.lower()
...
xvel = np.gradient(xpos_smooth)
yvel = np.gradient(ypos_smooth)
...
if is_tongue:
    xvel[np.isnan(xvel)] = 0
    yvel[np.isnan(yvel)] = 0
```
```python
interpolated = np.interp(time_axis, aligned_ft, spd)
speed[:, trial_idx] = interpolated.astype(np.float32)
```

iii. The trajectory shows the AI was trying to avoid a degenerate all-one discretization by turning "not visible" tongue periods into zeros before thresholding. The notes characterize the result as tongue-visible-or-moving versus not.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The code computes the session median of the aligned tongue-speed matrix, then binarizes it as `< threshold` versus `>= threshold`. If the threshold is zero or nearly zero, it forces a tiny positive epsilon so exact zeros fall into class `0`. There is no explicit class `2` for not visible.

ii. <Code snippets>
```python
tongue_thresh = np.nanpercentile(tongue_speed_valid, 50)
tongue_disc = discretize_time_series(tongue_speed_valid, tongue_thresh)
```
```python
def discretize_time_series(values, threshold):
    if threshold < 1e-10:
        threshold = 1e-10
    return (values >= threshold).astype(np.float32)
```

iii. The trajectory explicitly justifies the epsilon hack by noting that the median tongue speed was often zero, which otherwise would have made every non-negative value class `1`.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The code estimates one session-wide video offset, subtracts it and the trial go cue from the frame times, then interpolates tongue speed onto the same `time_axis` used for neural data. If frame times are missing, it fabricates a 400 Hz time base and uses a hard-coded `0.5 s` offset.

ii. <Code snippets>
```python
vidshift = find_video_offset(data, fmt)
...
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
interpolated = np.interp(time_axis, aligned_ft, spd)
```
```python
return 0.5  # default offset
```

iii. The notes justify the alignment by referencing `findVideoOffset.m`, but the fallback behavior is justified pragmatically rather than from the reference processing.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom-camera `traj` stream for the feature `top_paw`, along with its `frameTimes`, `ts` coordinates, go-cue times, and the session video offset.

ii. <Code snippets>
```python
paw_speed = extract_velocity_from_traj(
    data, fmt, view=1, feat_name='top_paw',
    ntrials=ntrials, align_times=align_times,
    time_axis=time_axis, vidshift=vidshift
)
```

iii. The notes justify `top_paw` as the intended paw feature from the bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The paw pipeline reuses the generic velocity extractor: position smoothing with a 21-sample causal Gaussian, plain `np.gradient` velocities, subtraction of the median baseline velocity, nearest-neighbor filling of NaNs, and linear interpolation to the neural time axis.

ii. <Code snippets>
```python
if not is_tongue:
    xpos_smooth = causal_gaussian_smooth(xpos, 21, 'reflect')
    ypos_smooth = causal_gaussian_smooth(ypos, 21, 'reflect')
...
xvel = np.gradient(xpos_smooth)
yvel = np.gradient(ypos_smooth)
```
```python
if not is_tongue:
    xvel = xvel - np.nanmedian(xvel)
    yvel = yvel - np.nanmedian(yvel)
...
spd = np.sqrt(xvel**2 + yvel**2)
```

iii. The notes justify this by analogy to the reference velocity code, but they do not justify the added baseline subtraction or omission of visibility-based masking.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is thresholded with the same binary median split used for tongue velocity: `< 50th percentile` becomes `0`, `>= 50th percentile` becomes `1`, and no explicit `not visible` class is emitted.

ii. <Code snippets>
```python
paw_thresh = np.nanpercentile(paw_speed_valid, 50)
paw_disc = discretize_time_series(paw_speed_valid, paw_thresh)
```

iii. The notes justify a median split because the decoder specification asked for a per-session 50th percentile threshold, but they do not justify omitting the requested third class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw frame times are offset-corrected and go-cue-aligned in the same way as tongue frame times, then linearly interpolated to the shared neural `time_axis`.

ii. <Code snippets>
```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
...
interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. The notes justify using the same video-alignment machinery for all camera-derived streams.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<animal>_<date>.mat` file. The code unwraps several possible MATLAB layouts and builds a Python list of per-trial motion-energy vectors.

ii. <Code snippets>
```python
def load_motion_energy(dirpath, animal, date):
    me_file = os.path.join(dirpath, f'motionEnergy_{animal}_{date}.mat')
    ...
    me_raw = me_mat['me']
```
```python
for i in range(me_data.shape[0]):
    elem = me_data[i, 0] if me_data.ndim == 2 else me_data[i]
    trial_me.append(elem.flatten().astype(np.float64))
```

iii. The notes justify the separate loader because motion energy is stored beside the session file and appears in multiple layouts.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code aligns motion-energy traces to side-camera frame times, interpolates them to the neural time axis, and fills missing values with nearest neighbors or zeros if an entire trial is missing. It does not leave missing video as a separate category.

ii. <Code snippets>
```python
me_interp[:, trial_idx] = np.interp(
    time_axis, aligned_ft[:n_frames], me_trial[:n_frames]
).astype(np.float32)
```
```python
if np.any(nan_mask) and not np.all(nan_mask):
    ...
elif np.all(nan_mask):
    col[:] = 0.0
```

iii. The notes justify interpolation by analogy to `loadMotionEnergy.m` and by the desire to match the neural time base.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded with a binary session-median split into `low` and `high`. There is no explicit `no video` category in the output encoding.

ii. <Code snippets>
```python
me_thresh_50 = np.nanpercentile(me_valid, 50)
me_disc = discretize_time_series(me_valid, me_thresh_50)
```

iii. The notes justify the median split from the task specification, but give no separate justification for removing the third category.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses side-camera frame times, subtracts the session video offset and trial go cue, and interpolates the trace to the shared `time_axis`.

ii. <Code snippets>
```python
ts, frame_times, _ = get_traj_data(data, fmt, 0, trial_idx)  # side cam
...
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
...
np.interp(time_axis, aligned_ft[:n_frames], me_trial[:n_frames])
```

iii. The notes justify this as the same synchronization scheme used for other camera-derived signals.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed video timing is handled by fabricating a 400 Hz frame clock and using a default `0.5 s` offset. Missing tongue velocities are turned into zeros; remaining NaNs in velocity or motion energy are filled by nearest neighbors; trials with all-missing motion energy become all-zero traces; and behavioral trials beyond the recording are dropped.

ii. <Code snippets>
```python
if len(frame_times) == 0 or np.all(np.isnan(frame_times)):
    frame_times = np.arange(n_frames) / 400.0
    ft_offset = 0.5
```
```python
if is_tongue:
    xvel[np.isnan(xvel)] = 0
    yvel[np.isnan(yvel)] = 0
```
```python
for idx in np.where(nan_mask)[0]:
    nearest = valid[np.argmin(np.abs(valid - idx))]
    spd[idx] = spd[nearest]
...
elif np.all(nan_mask):
    col[:] = 0.0
```

iii. The trajectory shows the AI saw the tongue visibility problem as something to repair numerically so the median threshold would work. The notes justify the recording-end cut as a fix for sessions with late all-zero neural data.

## 11-a. What are the most time-consuming steps of the code?

i. The AI's notes identify spike binning and per-trial extraction of tongue, paw, and motion-energy signals as the expensive steps. The full-run log reports separate timings for each of those stages per session.

ii. <Code snippets>
```python
print(f"    Spike binning: {time.time()-t1:.1f}s")
...
print(f"    Tongue velocity: {time.time()-t2:.1f}s")
...
print(f"    Paw velocity: {time.time()-t3:.1f}s")
...
print(f"    Motion energy: {time.time()-t4:.1f}s")
```

iii. In the notes, the AI estimated about 7-9 seconds per session and explicitly called out spike binning and DLC extraction loops as the main costs.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI noted that spike binning still loops over neurons and trials, while DLC extraction still loops over trials. It treated the camera-side loops as harder to vectorize because frame counts vary by trial.

ii. <Code snippets>
```python
for i, clu in enumerate(clusters):
    ...
    for j in range(ntrials):
        ...
        counts = np.histogram(aligned_times, bins=edges)[0]
```
```python
for trial_idx in range(ntrials):
    ...
```

iii. The notes explicitly say spike binning loops over neurons and trials, and say the DLC loops were kept because of variable-length frame sequences.

## 11-c. What processing does the code repeat multiple times?

i. The code repeatedly re-reads trajectory fields and repeats similar interpolation-and-fill logic separately for tongue, paw, and motion energy. It also rebuilds the identical `input_data` array for every trial and computes neural rates for all trials before discarding invalid ones.

ii. <Code snippets>
```python
tongue_speed = extract_velocity_from_traj(...)
paw_speed = extract_velocity_from_traj(...)
me_interp = interpolate_motion_energy(...)
```
```python
input_data = time_axis.astype(np.float32).reshape(1, -1)
input_trials.append(input_data)
```
```python
trialdat = bin_and_smooth_spikes(all_clusters, ntrials, align_times, time_axis, edges)
...
trialdat_valid = trialdat[:, :, valid_trial_indices]
```

iii. There is little explicit justification in the notes beyond implementation simplicity and keeping a single reusable extractor for video-derived signals.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores intermediate values that are discarded: `L_valid`, `miss_valid`, `align_times_valid`, `me_thresh`, and full neural data for invalid trials before subselecting valid ones. It also loads trajectory information repeatedly and optionally generates diagnostic plots that are not part of the final dataset.

ii. <Code snippets>
```python
miss_valid = miss[valid_trial_indices]
L_valid = L[valid_trial_indices]
align_times_valid = align_times[valid_trial_indices]
...
me_data, me_thresh = load_motion_energy(dirpath, animal, date)
```
```python
trialdat = bin_and_smooth_spikes(all_clusters, ntrials, align_times, time_axis, edges)
...
trialdat_valid = trialdat[:, :, valid_trial_indices]
```
```python
if show_processing and session_idx < 2:
    generate_processing_plots(...)
```

iii. The notes do not explicitly justify these extra computations. The closest justification is that the AI was prioritizing validation and debugging while iterating on the conversion.
