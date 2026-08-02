# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes a session table for 44 electrophysiology sessions, then iterates that table in `main()`. For each session it opens one `data_structure_<animal>_<date>.mat` file with either `h5py` or `scipy.io.loadmat`, reads behavior from `obj.bp`, spikes from `obj.clu`, video from `obj.traj`, and motion energy from a separate `motionEnergy_<animal>_<date>.mat` file.

ii. <Code snippets>

```python
EPHYS_SESSIONS = [
    ('data/Ephys_Behavior', 'JEB6', '2021-04-18', [2]),
    ...
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-11-03', [1]),
]
```

```python
for sess_idx, (dirpath, animal, date, probes) in enumerate(sessions):
    result = process_session(
        dirpath, animal, date, probes, time_axis, edges,
        show_processing=args.show_processing, session_idx=sess_idx
    )
```

```python
filepath = os.path.join(dirpath, f"data_structure_{session_id}.mat")
data, fmt = load_mat_file(filepath)
...
me_data, me_thresh = load_motion_energy(dirpath, animal, date)
```

iii. <Justification>

The notes say the session list came from the MATLAB loader scripts and that only the two electrophysiology directories were used because the inhibition-only datasets have no neural data. The notes also state that the three randomized-delay files not present in the MATLAB loaders were excluded.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the `animal` field in each hard-coded session tuple. After processing, the script builds `subjects` as the sorted unique animal names and `subject_idx` as a per-session index into that list.

ii. <Code snippets>

```python
all_animals.append(animal)
...
unique_subjects = sorted(set(all_animals))
subject_idx = np.array([unique_subjects.index(a) for a in all_animals])
```

iii. <Justification>

The notes describe subjects as the unique animal IDs present in the loader-script session list.

## 1-c. How are the data split into sessions?

i. Each `(directory, animal, date, probe_numbers)` tuple is treated as one session. `process_session()` loads one `.mat` file per tuple and returns one session entry for `neural`, `input`, `output`, and metadata.

ii. <Code snippets>

```python
def process_session(dirpath, animal, date, probe_nums, time_axis, edges, ...):
    session_id = f"{animal}_{date}"
    filepath = os.path.join(dirpath, f"data_structure_{session_id}.mat")
```

```python
all_neural.append(result['neural'])
all_input.append(result['input'])
all_output.append(result['output'])
session_info.append({
    'animal': animal,
    'date': date,
    ...
})
```

iii. <Justification>

The notes say the session boundaries come directly from the MATLAB `loadANM_*_ALMVideo` loader definitions, including probe assignments and excluded randomized-delay dates.

## 1-d. How are the data split into trials?

i. Trials are indexed by `obj.bp.Ntrials`. Behavioral arrays are read as per-trial vectors, spike data are binned into a `(time, neuron, trial)` array with one slot per behavioral trial, and then only `valid_trial_indices` are retained. Each retained trial becomes one `(neurons, time)` matrix plus one input array and one output array.

ii. <Code snippets>

```python
ntrials = get_ntrials(data, fmt)
...
trialdat = np.zeros((n_time, n_neurons, ntrials), dtype=np.float32)
for j in range(ntrials):
    trial_num = j + 1
    spike_mask = trial == trial_num
```

```python
valid_trial_indices = np.where(valid_trials)[0]
trialdat_valid = trialdat[:, :, valid_trial_indices]
...
for t_idx in range(len(valid_trial_indices)):
    neural_trials.append(trialdat_valid[:, :, t_idx].T.astype(np.float32))
```

iii. <Justification>

The notes say the agent followed the MATLAB convention that trials are stored in Bpod order and then subset by logical trial filters.

## 1-e. How are trials filtered based on quality controls?

i. The script keeps trials with `stim.enable == 0`, `early == 0`, and valid positive go-cue times. It also drops trials after the last trial index present in any spike train and skips sessions with fewer than two retained trials. It does not explicitly remove ignore (`no`) trials.

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
valid_trial_indices = valid_trial_indices[valid_trial_indices < max_spike_trial]
```

```python
if len(valid_trial_indices) < 2:
    print(f"    SKIP: Too few valid trials")
    return None
```

iii. <Justification>

The notes say the intended rule was “exclude stim.enable and early, keep hit/miss/no,” and later document the extra “beyond recording” filter as a fix for sessions where the behavioral session outlasted the spike recording.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from `obj.clu{probe}(cluster).trialtm`, `obj.clu{probe}(cluster).trial`, and cluster `quality`, restricted to the probe numbers listed for each session. The code does not use waveform fields or `tm`.

ii. <Code snippets>

```python
trialtm = f[trialtm_ref][:].flatten()
trial = f[trial_ref][:].flatten().astype(int)
clusters.append({
    'trialtm': trialtm,
    'trial': trial,
    'quality': quality,
})
```

```python
for p in probe_nums:
    clusters = get_clusters(data, fmt, p - 1)
    all_clusters.extend(clusters)
```

iii. <Justification>

The notes map `obj.clu` spike times to the neural output and explicitly say only ALM probe(s) from the loader scripts are used.

## 2-b. How is the `neural` data processed?

i. For each neuron and trial, spike times are aligned by subtracting the trial’s go-cue time, histogrammed into `-2.5:0.01:2.5` bins, converted to firing rates by dividing by `DT`, and smoothed with a causal Gaussian filter of width `N=15` and reflect padding. The final per-trial output is stored as `(neurons, time)`.

ii. <Code snippets>

```python
aligned_times = trialtm[spike_mask] - align_times[j]
counts = np.histogram(aligned_times, bins=edges)[0]
rate = counts.astype(np.float64) / DT
smoothed = causal_gaussian_smooth(rate, SMOOTH_N, SMOOTH_BC)
trialdat[:, i, j] = smoothed.astype(np.float32)
```

```python
DT = 1.0 / 100
SMOOTH_N = 15
SMOOTH_BC = 'reflect'
```

iii. <Justification>

The notes say this was meant to match `alignSpikes.m`, `getSeq.m`, and `mySmooth.m`, and repeatedly justify the `10 ms`, `N=15`, `reflect` choices from the tutorial code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters whose quality string is in `{'garbage','gabrga','noisy','real?'}` are excluded at load time. After binning, neurons with mean firing rate `<= 1 Hz` are dropped using the mean over all time bins and all trials in `trialdat`. Sessions with fewer than 10 neurons before or after this filter are skipped.

ii. <Code snippets>

```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
...
if quality in EXCLUDE_QUALITIES:
    continue
```

```python
mean_frs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
keep = mean_frs > LOW_FR_THRESHOLD
trialdat_filtered = trialdat[:, keep, :]
```

```python
if len(all_clusters) < MIN_UNITS:
    return None
...
if n_neurons < MIN_UNITS:
    return None
```

iii. <Justification>

The notes claim this matches `findClusters.m` and `removeLowFRClusters.m`, and cite the paper’s `>1 Hz` and `>=10 units/session` criteria.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to go-cue onset by subtracting `obj.bp.ev.goCue(trial)` from each spike’s trial time and then binning into a window from `-2.5 s` to `+2.5 s`.

ii. <Code snippets>

```python
ALIGN_EVENT = 'goCue'
TMIN = -2.5
TMAX = 2.5
```

```python
align_times = get_event_times(data, fmt, ALIGN_EVENT)
aligned_times = trialtm[spike_mask] - align_times[j]
```

iii. <Justification>

The notes state alignment to `goCue` was taken directly from `WorkingWithDataObjs.m` and the decoder instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins (`DT = 1/100`). No additional rebinning is applied after that initial spike histogramming; smoothing is applied on the 10 ms grid.

ii. <Code snippets>

```python
DT = 1.0 / 100  # 10 ms time bins
edges = np.arange(TMIN, TMAX + DT/2, DT)
time_axis = edges[:-1] + DT / 2
```

iii. <Justification>

The notes acknowledge ambiguity because another default file uses 5 ms, but say the agent chose 10 ms from `WorkingWithDataObjs.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not derived from a trial-specific raw behavioral variable beyond the decision to align everything to go cue. It is generated from the configured constants `TMIN`, `TMAX`, and `DT` to make a common relative time axis.

ii. <Code snippets>

```python
def compute_time_axis():
    edges = np.arange(TMIN, TMAX + DT/2, DT)
    time_axis = edges[:-1] + DT / 2
    return time_axis, edges
```

```python
input_data = time_axis.astype(np.float32).reshape(1, -1)
```

iii. <Justification>

The notes explicitly say the decoder input is “same time axis for all trials: tmin:dt:tmax centered.”

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script computes bin centers from the configured window and bin size, converts them to `float32`, reshapes them to `(1, time)`, and reuses the exact same array for every retained trial.

ii. <Code snippets>

```python
edges = np.arange(TMIN, TMAX + DT/2, DT)
time_axis = edges[:-1] + DT / 2
```

```python
for t_idx in range(len(valid_trial_indices)):
    input_data = time_axis.astype(np.float32).reshape(1, -1)
    input_trials.append(input_data)
```

iii. <Justification>

The notes justify this as the required “time from go cue onset” decoder input.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the same `time_axis` used to bin neural data, so each input time bin corresponds exactly to the neural bin centers relative to go cue.

ii. <Code snippets>

```python
time_axis, edges = compute_time_axis()
...
trialdat = bin_and_smooth_spikes(all_clusters, ntrials, align_times, time_axis, edges)
...
input_data = time_axis.astype(np.float32).reshape(1, -1)
```

iii. <Justification>

The notes say the time axis was chosen specifically so all neural, video, and motion-energy streams share a common alignment.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The script derives this output from `obj.bp.R` alone. It loads `L` too, but does not use it in the final computation.

ii. <Code snippets>

```python
R = get_bp_field(data, fmt, 'R').astype(bool)
L = get_bp_field(data, fmt, 'L').astype(bool)
...
lick_direction = R_valid.astype(np.float32)
```

iii. <Justification>

The notes show the agent debated whether to use actual lick side on error trials, but ultimately chose to treat `R/L` as the per-trial “lick direction” or trial type.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. After trial filtering, the script casts `R_valid` to `float32`, then writes that scalar into every time bin of the trial’s output row. So the signal is per-trial constant rather than event-based.

ii. <Code snippets>

```python
lick_direction = R_valid.astype(np.float32)
...
out[0, :] = int(lick_direction[t_idx])
```

iii. <Justification>

The notes explicitly say “keep it simple: `R=1, L=0` as the instruction/stimulus direction.”

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`.

ii. <Code snippets>

```python
autowater = get_bp_field(data, fmt, 'autowater').astype(bool)
...
behavioral_context = (~autowater_valid).astype(np.float32)
```

iii. <Justification>

The notes map `autowater=1` to water-cued and `autowater=0` to delayed response.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The script inverts the Boolean `autowater` flag so that `WC=0` and `DR=1`, then repeats that scalar across all time bins of each trial.

ii. <Code snippets>

```python
behavioral_context = (~autowater_valid).astype(np.float32)
...
out[1, :] = int(behavioral_context[t_idx])
```

iii. <Justification>

The notes say this coding was chosen directly from the decoder task specification.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`. The code also loads `miss`, but does not use it explicitly in the final assignment.

ii. <Code snippets>

```python
hit = get_bp_field(data, fmt, 'hit').astype(bool)
miss = get_bp_field(data, fmt, 'miss').astype(bool)
...
outcome = hit_valid.astype(np.float32)
```

iii. <Justification>

The notes say the intended mapping was `hit -> 1`, `miss/no -> 0`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The script converts `hit_valid` to `float32` and repeats that scalar across all time bins in the trial output. Any retained non-hit trial becomes `0`.

ii. <Code snippets>

```python
outcome = hit_valid.astype(np.float32)
...
out[2, :] = int(outcome[t_idx])
```

iii. <Justification>

The notes justify this as the required “incorrect = 0, correct = 1” output, with ignore trials implicitly falling into the incorrect class if retained.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It is derived from the side-camera `obj.traj` data for the DeepLabCut feature named `'tongue'`, using `ts` and `frameTimes`.

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
feat_idx = feat_names.index(feat_name)
```

iii. <Justification>

The notes first considered using jaw as a proxy, then decided to use the tracked tongue feature because the task explicitly asked for tongue velocity.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the script reads tongue x/y positions, aligns frame times to go cue, leaves tongue positions unsmoothed, computes x and y velocity by `np.gradient`, replaces NaN tongue velocities with zero, takes speed magnitude `sqrt(xvel^2 + yvel^2)`, fills missing values, and interpolates onto the neural time axis.

ii. <Code snippets>

```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
...
xvel = np.gradient(xpos_smooth)
yvel = np.gradient(ypos_smooth)
...
xvel[np.isnan(xvel)] = 0
yvel[np.isnan(yvel)] = 0
spd = np.sqrt(xvel**2 + yvel**2)
interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. <Justification>

The notes say this was intended to mirror `findVelocity.m` behavior for tongue, especially the “if not visible, set velocity to 0” rule.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The script computes a per-session 50th percentile threshold across all retained timepoints from all retained trials, then assigns `0` below threshold and `1` at or above threshold. If the threshold is near zero it is replaced with `1e-10` so exact zeros stay in class `0`.

ii. <Code snippets>

```python
tongue_thresh = np.nanpercentile(tongue_speed_valid, 50)
tongue_disc = discretize_time_series(tongue_speed_valid, tongue_thresh)
```

```python
if threshold < 1e-10:
    threshold = 1e-10
return (values >= threshold).astype(np.float32)
```

iii. <Justification>

The notes state the 50th-percentile split was required by the decoder task rather than copied from the paper’s manual motion thresholding procedure.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. It is aligned by subtracting the video offset and the trial-specific go-cue time from video frame times, then interpolating the resulting tongue speed to the same `time_axis` as neural data.

ii. <Code snippets>

```python
vidshift = find_video_offset(data, fmt)
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
interpolated = np.interp(time_axis, aligned_ft, spd)
speed[:, trial_idx] = interpolated.astype(np.float32)
```

iii. <Justification>

The notes cite `findVideoOffset.m`, `findPosition.m`, and the shared go-cue time axis as the basis for this alignment.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom-camera `obj.traj` feature `'top_paw'`, again using trajectory `ts` and `frameTimes`.

ii. <Code snippets>

```python
paw_speed = extract_velocity_from_traj(
    data, fmt, view=1, feat_name='top_paw',
    ntrials=ntrials, align_times=align_times,
    time_axis=time_axis, vidshift=vidshift
)
```

iii. <Justification>

The notes explicitly say they chose bottom-view `top_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The script extracts x/y position for `top_paw`, smooths position with a causal Gaussian (`N=21`), computes velocity with `np.gradient`, subtracts the median x and y velocity as baseline, converts to speed magnitude, fills missing values with nearest neighbors, and interpolates to the common neural time axis.

ii. <Code snippets>

```python
if not is_tongue:
    xpos_smooth = causal_gaussian_smooth(xpos, 21, 'reflect')
    ypos_smooth = causal_gaussian_smooth(ypos, 21, 'reflect')
...
xvel = np.gradient(xpos_smooth)
yvel = np.gradient(ypos_smooth)
xvel = xvel - np.nanmedian(xvel)
yvel = yvel - np.nanmedian(yvel)
spd = np.sqrt(xvel**2 + yvel**2)
```

iii. <Justification>

The notes claim this follows the reference velocity extraction, but the code actually reduces paw velocity to scalar speed rather than preserving x/y components.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It uses the per-session median over all retained paw-speed samples, with values below the median mapped to `0` and values at or above mapped to `1`.

ii. <Code snippets>

```python
paw_thresh = np.nanpercentile(paw_speed_valid, 50)
paw_disc = discretize_time_series(paw_speed_valid, paw_thresh)
```

iii. <Justification>

The notes say this was chosen to satisfy the decoder task’s required binarization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Exactly like tongue velocity: bottom-camera frame times are shifted by the video offset and go-cue time, then the paw speed is interpolated to the neural bin centers.

ii. <Code snippets>

```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
interpolated = np.interp(time_axis, aligned_ft, spd)
speed[:, trial_idx] = interpolated.astype(np.float32)
```

iii. <Justification>

The notes justify this by reference to the shared `time_axis` and the video-offset logic.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<animal>_<date>.mat` file, specifically the `me` structure or cell array, usually `me.data`.

ii. <Code snippets>

```python
me_mat = scipy.io.loadmat(me_file, squeeze_me=False)
me_raw = me_mat['me']
...
me_data = me_raw['data'][0, 0]
```

iii. <Justification>

The notes repeatedly say motion energy comes from the separate motion-energy files and not from the DLC trajectories.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The script normalizes several possible MATLAB file layouts into a list of per-trial arrays, then interpolates each trial’s motion-energy trace onto the neural time axis using side-camera frame times aligned to go cue. Missing values are filled with nearest neighbors, and fully missing trials become zeros.

ii. <Code snippets>

```python
for i in range(me_data.shape[0]):
    elem = me_data[i, 0] if me_data.ndim == 2 else me_data[i]
    trial_me.append(elem.flatten().astype(np.float64))
```

```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
me_interp[:, trial_idx] = np.interp(
    time_axis, aligned_ft[:n_frames], me_trial[:n_frames]
).astype(np.float32)
```

iii. <Justification>

The notes say this was meant to match `loadMotionEnergy.m` except for the downstream decoder-specific discretization.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The script ignores the paper’s manual `me.moveThresh` for decoder output construction and instead applies a per-session median split over all retained motion-energy samples.

ii. <Code snippets>

```python
me_data, me_thresh = load_motion_energy(dirpath, animal, date)
...
me_thresh_50 = np.nanpercentile(me_valid, 50)
me_disc = discretize_time_series(me_valid, me_thresh_50)
```

iii. <Justification>

The notes explain this as a deliberate deviation required by the decoder task, even though the reference paper/code use a manually selected bimodal threshold for movement.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The motion-energy trace is aligned by using the side-camera frame times, subtracting video offset and go-cue time, and interpolating onto the same bin centers as neural data.

ii. <Code snippets>

```python
ts, frame_times, _ = get_traj_data(data, fmt, 0, trial_idx)
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
me_interp[:, trial_idx] = np.interp(
    time_axis, aligned_ft[:n_frames], me_trial[:n_frames]
).astype(np.float32)
```

iii. <Justification>

The notes cite `loadMotionEnergy.m` and the shared alignment convention.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses a collection of fallback rules: missing `stim.enable` becomes all zeros; missing motion-energy files become all-zero motion energy; missing or all-NaN frame times are replaced with synthetic `400 Hz` frame times and a `0.5 s` offset; missing tongue velocities become zero; other missing kinematic or motion-energy samples are nearest-filled; unknown motion-energy formats are skipped with a warning; late behavioral trials beyond the last spike trial are excluded; and sessions with too few trials or neurons are skipped.

ii. <Code snippets>

```python
try:
    stim_enable = get_bp_field(data, fmt, 'stim.enable').astype(bool)
except Exception:
    stim_enable = np.zeros(ntrials, dtype=bool)
```

```python
if len(frame_times) == 0 or np.all(np.isnan(frame_times)):
    frame_times = np.arange(n_frames) / 400.0
    ft_offset = 0.5
```

```python
if is_tongue:
    xvel[np.isnan(xvel)] = 0
    yvel[np.isnan(yvel)] = 0
...
speed[:, nan_cols] = 0.0
```

```python
if me_data is not None:
    ...
else:
    me_valid = np.zeros((len(time_axis), len(valid_trial_indices)), dtype=np.float32)
```

iii. <Justification>

The notes describe these as “sensible defaults” and explicitly document the added beyond-recording filter after verification found zero-neural trials in two sessions.

## 11-a. What are the most time-consuming steps of the code?

i. The most expensive steps are the nested-loop spike binning/smoothing, then the per-trial tongue and paw velocity extraction and interpolation, followed by motion-energy interpolation.

ii. <Code snippets>

```python
t1 = time.time()
trialdat = bin_and_smooth_spikes(...)
print(f"    Spike binning: {time.time()-t1:.1f}s")
```

```python
t2 = time.time()
tongue_speed = extract_velocity_from_traj(...)
print(f"    Tongue velocity: {time.time()-t2:.1f}s")
...
t3 = time.time()
paw_speed = extract_velocity_from_traj(...)
...
t4 = time.time()
me_interp = interpolate_motion_energy(...)
```

iii. <Justification>

The notes identify spike binning and DLC extraction as the main bottlenecks and record per-session runtimes around 6 to 9 seconds.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunities are the neuron-by-trial spike loop in `bin_and_smooth_spikes`, the repeated nearest-fill loops over NaNs for kinematics and motion energy, the per-trial construction of identical `input_data`, and the per-trial assembly of `neural_trials` and `output_trials`.

ii. <Code snippets>

```python
for i, clu in enumerate(clusters):
    ...
    for j in range(ntrials):
        spike_mask = trial == trial_num
        ...
        counts = np.histogram(aligned_times, bins=edges)[0]
```

```python
for idx in np.where(nan_mask)[0]:
    nearest = valid[np.argmin(np.abs(valid - idx))]
    spd[idx] = spd[nearest]
```

```python
for t_idx in range(len(valid_trial_indices)):
    input_data = time_axis.astype(np.float32).reshape(1, -1)
    input_trials.append(input_data)
```

iii. <Justification>

The notes mention the spike loop as a known inefficiency, but the script still performs most operations in Python loops.

## 11-c. What processing does the code repeat multiple times?

i. The code repeatedly computes aligned video times and nearest-fills NaNs separately for tongue, paw, and motion energy; repeatedly converts the same `time_axis` to `float32` once per trial; and recomputes frame-time fallbacks independently in the kinematics and motion-energy paths.

ii. <Code snippets>

```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
...
interpolated = np.interp(time_axis, aligned_ft, spd)
```

```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
me_interp[:, trial_idx] = np.interp(
    time_axis, aligned_ft[:n_frames], me_trial[:n_frames]
).astype(np.float32)
```

```python
input_data = time_axis.astype(np.float32).reshape(1, -1)
input_trials.append(input_data)
```

iii. <Justification>

The notes claim the code is efficient enough for the dataset size, but they do not describe any deduplication of these repeated per-trial operations.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script bins and smooths spikes for all behavioral trials before discarding invalid trials; loads `miss`, `L`, `align_times_valid`, `keep_mask`, and `me_thresh` without using them downstream; computes optional processing plots that are not part of the saved dataset; and stores per-trial outputs as full time-varying constant rows for the scalar labels instead of compact per-trial values.

ii. <Code snippets>

```python
trialdat = bin_and_smooth_spikes(all_clusters, ntrials, align_times, time_axis, edges)
...
trialdat_valid = trialdat[:, :, valid_trial_indices]
```

```python
miss_valid = miss[valid_trial_indices]
L_valid = L[valid_trial_indices]
align_times_valid = align_times[valid_trial_indices]
...
me_data, me_thresh = load_motion_energy(dirpath, animal, date)
```

```python
out[0, :] = int(lick_direction[t_idx])
out[1, :] = int(behavioral_context[t_idx])
out[2, :] = int(outcome[t_idx])
```

iii. <Justification>

The notes focus on runtime sufficiency rather than minimizing discarded work, so these redundancies are mostly visible from the code rather than from explicit agent commentary.
