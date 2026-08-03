# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a 44-session `EPHYS_SESSIONS` list of `(directory, animal, date, probes)` tuples, then iterates that list in `main()`. For each session it opens `data_structure_<animal>_<date>.mat` with `load_mat_file()`, which supports both MATLAB v7.3 HDF5 files and older v5 files. Motion energy is loaded separately from `motionEnergy_<animal>_<date>.mat`.

ii.
```python
EPHYS_SESSIONS = [
    ('data/Ephys_Behavior', 'JEB6', '2021-04-18', [2]),
    ...
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-11-03', [1]),
]
```
```python
filepath = os.path.join(dirpath, f"data_structure_{session_id}.mat")
data, fmt = load_mat_file(filepath)
...
me_data, me_thresh = load_motion_energy(dirpath, animal, date)
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

iii. In `CONVERSION_NOTES.md`, the AI says probe assignments came from the author loader scripts and that the shared data mix HDF5 v7.3 and MATLAB v5 files, so both readers were needed.

## 1-b. How are the data split into subjects (mice)?

i. The AI uses the `animal` field from each session tuple as the subject id. After processing, it builds `subjects` as the sorted unique animal names and `subject_idx` as one index per session.

ii.
```python
all_animals.append(animal)
...
unique_subjects = sorted(set(all_animals))
subject_idx = np.array([unique_subjects.index(a) for a in all_animals])
```

iii. In the notes, the AI states that the 44 sessions span 14 unique animals and keeps those loader-script animal names as the subject identifiers.

## 1-c. How are the data split into sessions?

i. One session is one tuple in `EPHYS_SESSIONS`, one `data_structure_*.mat` file, and one output element in `neural`, `input`, and `output`. The fixed-delay and randomized-delay folders are treated uniformly because both are represented in the same explicit session list.

ii.
```python
for sess_idx, (dirpath, animal, date, probes) in enumerate(sessions):
    result = process_session(
        dirpath, animal, date, probes, time_axis, edges,
        show_processing=args.show_processing, session_idx=sess_idx
    )
```

iii. `CONVERSION_NOTES.md` says only the two ephys folders are used and that the three extra randomized-delay files not present in the author loader scripts are excluded.

## 1-d. How are the data split into trials?

i. The AI uses `bp.Ntrials` as the session trial count, reads per-trial behavioral arrays of that length, and builds `valid_trial_indices` as the kept trial indices. Neural and video data are then subset by those same trial indices.

ii.
```python
ntrials = get_ntrials(data, fmt)
...
valid_trial_indices = np.where(valid_trials)[0]
...
trialdat_valid = trialdat[:, :, valid_trial_indices]
...
hit_valid = hit[valid_trial_indices]
```

iii. The notes describe `obj.bp` as the per-trial behavioral table and treat each session as a sequence of trials indexed directly by those arrays.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops photostimulation trials (`stim.enable`), early-lick trials (`early`), trials with missing or non-positive go cue times, and late behavioral trials that extend beyond the maximum trial index observed in spike data.

ii.
```python
valid_trials = ~stim_enable & ~early
valid_trials &= ~np.isnan(align_times) & (align_times > 0)
...
max_spike_trial = max(
    int(np.max(c['trial'])) for c in all_clusters if len(c['trial']) > 0
)
...
valid_trial_indices = valid_trial_indices[valid_trial_indices < max_spike_trial]
```

iii. In the notes, the AI cites the paper/code for excluding `stim.enable` and `early`, and later documents the extra “beyond recording” filter after finding zero-neural late trials in two sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural signal comes from cluster spike times and trial assignments in `obj.clu`, plus go cue times from `obj.bp.ev.goCue` for alignment. Cluster quality labels are used for filtering.

ii.
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

iii. `CONVERSION_NOTES.md` explicitly maps neural output to `obj.clu spike times`, quality filtering, and go-cue alignment.

## 2-b. How is the `neural` data processed?

i. The AI bins spikes into 10 ms bins from -2.5 s to +2.5 s around the go cue, converts counts to firing rates by dividing by `DT`, and smooths each trial trace with a custom causal Gaussian kernel of width `SMOOTH_N=15`.

ii.
```python
DT = 1.0 / 100  # 10 ms time bins
SMOOTH_N = 15  # causal Gaussian smoothing window
```
```python
counts = np.histogram(aligned_times, bins=edges)[0]
rate = counts.astype(np.float64) / DT
smoothed = causal_gaussian_smooth(rate, SMOOTH_N, SMOOTH_BC)
trialdat[:, i, j] = smoothed.astype(np.float32)
```

iii. In the notes, the AI says it chose 10 ms bins from `WorkingWithDataObjs.m` because it viewed the reference code as ambiguous between 10 ms and 5 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI excludes clusters whose lower-cased quality labels are in `{'garbage', 'gabrga', 'noisy', 'real?'}`, then removes neurons whose mean firing rate is not above 1 Hz. It also skips entire sessions with fewer than 10 units after filtering.

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR_THRESHOLD = 1.0
MIN_UNITS = 10
```
```python
if quality in EXCLUDE_QUALITIES:
    continue
...
mean_frs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
keep = mean_frs > LOW_FR_THRESHOLD
```

iii. The notes justify the quality labels from `findClusters.m`, the 1 Hz threshold from the paper, and the minimum-session-unit criterion from the paper’s session inclusion rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns each spike time by subtracting the per-trial go cue time before binning.

ii.
```python
aligned_times = trialtm[spike_mask] - align_times[j]
counts = np.histogram(aligned_times, bins=edges)[0]
```

iii. The notes identify `goCue` as the alignment event and cite `alignSpikes.m` as the matching reference step.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10 ms bins over a 5 s window, yielding 500 time bins. No later temporal rebinning is applied.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 100
...
edges = np.arange(TMIN, TMAX + DT/2, DT)
time_axis = edges[:-1] + DT / 2
```

iii. The AI’s notes say this 10 ms choice came from preferring `WorkingWithDataObjs.m` over the 5 ms setting in `getDefaultParams.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is synthetic rather than a raw stored stream. It is defined from the chosen alignment window and bin size around go cue onset.

ii.
```python
def compute_time_axis():
    edges = np.arange(TMIN, TMAX + DT/2, DT)
    time_axis = edges[:-1] + DT / 2
    return time_axis, edges
```

iii. The notes describe this as a decoder input defined on the same go-cue-centered time axis used for neural binning.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes bin edges from `TMIN`, `TMAX`, and `DT`, converts them to bin centers, and reuses that 1D time vector for every trial.

ii.
```python
edges = np.arange(TMIN, TMAX + DT/2, DT)
time_axis = edges[:-1] + DT / 2
...
input_data = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes frame this as a direct construction of the common go-cue time axis rather than a derived measurement.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the same bin grid used to bin spikes. The AI stores the time-bin centers as the input and bins neural spikes with the corresponding `edges`.

ii.
```python
time_axis, edges = compute_time_axis()
...
trialdat = bin_and_smooth_spikes(all_clusters, ntrials, align_times, time_axis, edges)
...
input_data = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes repeatedly describe a single shared time axis for neural, input, and behavioral outputs.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI loads `R`, `L`, `hit`, and `miss`, but the final lick-direction output is actually derived only from `R` after trial filtering.

ii.
```python
hit = get_bp_field(data, fmt, 'hit').astype(bool)
miss = get_bp_field(data, fmt, 'miss').astype(bool)
R = get_bp_field(data, fmt, 'R').astype(bool)
L = get_bp_field(data, fmt, 'L').astype(bool)
...
R_valid = R[valid_trial_indices]
...
lick_direction = R_valid.astype(np.float32)
```

iii. In the notes and trajectory, the AI justified this as using the “trial instruction/stimulus side” as lick direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. After filtering to valid trials, the AI encodes right-instructed trials as `1` and left-instructed trials as `0`. It does not invert miss trials and does not create a separate no-lick class.

ii.
```python
# Lick direction: R=1, L=0 (trial instruction/stimulus side)
lick_direction = R_valid.astype(np.float32)
```

iii. The notes show the AI consciously simplified lick direction to the instructed side because outcome was stored separately.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The behavioral-context output is derived from `obj.bp.autowater`.

ii.
```python
autowater = get_bp_field(data, fmt, 'autowater').astype(bool)
```

iii. The notes explicitly map `autowater=1` to WC and `autowater=0` to DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI relabels `autowater` trials as WC (`0`) and all other trials as DR (`1`).

ii.
```python
# Behavioral context: WC=0, DR=1
behavioral_context = (~autowater_valid).astype(np.float32)
```

iii. The notes justify this as a direct mapping from the task structure and the prompt’s label order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI loads both `hit` and `miss`, but the final outcome output is derived only from `hit` after trial filtering.

ii.
```python
hit = get_bp_field(data, fmt, 'hit').astype(bool)
miss = get_bp_field(data, fmt, 'miss').astype(bool)
...
hit_valid = hit[valid_trial_indices]
...
outcome = hit_valid.astype(np.float32)
```

iii. In the notes, the AI says it uses `correct(hit)=1` and collapses “incorrect(miss/no)=0`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI sets `outcome=1` for hit trials and `0` for all non-hit trials. It does not preserve a distinct ignore category.

ii.
```python
# Outcome: correct=1, incorrect=0
outcome = hit_valid.astype(np.float32)
```

iii. The notes justify this by treating misses and no-response trials as a single incorrect class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI derives tongue velocity from DeepLabCut trajectory data in `obj.traj`, using the side camera (`view=0`) feature named `'tongue'`, along with per-trial frame times, go cue times, and a computed video offset.

ii.
```python
def get_traj_data(data, fmt, view, trial_idx):
    ...
    return ts, frame_times, feat_names
```
```python
tongue_speed = extract_velocity_from_traj(
    data, fmt, view=0, feat_name='tongue',
    ntrials=ntrials, align_times=align_times,
    time_axis=time_axis, vidshift=vidshift
)
```

iii. In the trajectory, the AI explicitly debated using jaw as a proxy, then chose side-camera tongue because the task said “tongue velocity.”

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the AI extracts tongue x/y positions, synthesizes 400 Hz frame times if missing, aligns the video clock to go cue time, computes frame-to-frame gradients of x and y, converts NaN tongue velocities to zero, interpolates onto the neural time axis, fills remaining gaps with nearest-neighbor values, and then median-splits the session.

ii.
```python
if len(frame_times) == 0 or np.all(np.isnan(frame_times)):
    n_frames = ts.shape[2] if fmt == 'h5' else ts.shape[0]
    frame_times = np.arange(n_frames) / 400.0
    ft_offset = 0.5
...
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
...
xvel = np.gradient(xpos_smooth)
yvel = np.gradient(ypos_smooth)
...
xvel[np.isnan(xvel)] = 0
yvel[np.isnan(yvel)] = 0
...
interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. The notes and trajectory justify this mainly as a pragmatic way to make tongue velocity usable despite poor visibility and a zero-heavy median.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI uses the 50th percentile of all valid tongue-speed samples within a session as the threshold. If that threshold is effectively zero, it forces a tiny positive threshold so exact zeros map to class 0. The result is binary only.

ii.
```python
tongue_thresh = np.nanpercentile(tongue_speed_valid, 50)
...
def discretize_time_series(values, threshold):
    if threshold < 1e-10:
        threshold = 1e-10
    return (values >= threshold).astype(np.float32)
```

iii. The trajectory shows the AI added the tiny-threshold rule after noticing that tongue visibility made the session median frequently zero.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI computes a session-wide video offset from bitcode timing when possible, otherwise falls back to `0.5` s, subtracts that offset and the per-trial go cue from video frame times, and linearly interpolates tongue speed onto the neural time axis.

ii.
```python
vidshift = find_video_offset(data, fmt)
...
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
...
interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. The notes state that video had to be synchronized to the neural/behavior clock using bitcode timing or a 0.5 s default offset.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The AI derives paw velocity from DeepLabCut trajectory data in `obj.traj`, specifically the bottom-camera (`view=1`) feature `'top_paw'`, plus frame times, go cue times, and video offset.

ii.
```python
paw_speed = extract_velocity_from_traj(
    data, fmt, view=1, feat_name='top_paw',
    ntrials=ntrials, align_times=align_times,
    time_axis=time_axis, vidshift=vidshift
)
```

iii. The notes identify `top_paw` as the paw feature used from the bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The paw uses the same extraction helper as tongue, except non-tongue features are smoothed with a 21-sample causal Gaussian, their x/y gradients have baseline medians subtracted, and the resulting speed is interpolated and nearest-filled on the neural grid before sessionwise median splitting.

ii.
```python
if not is_tongue:
    xpos_smooth = causal_gaussian_smooth(xpos, 21, 'reflect')
    ypos_smooth = causal_gaussian_smooth(ypos, 21, 'reflect')
...
xvel = np.gradient(xpos_smooth)
yvel = np.gradient(ypos_smooth)
...
xvel = xvel - np.nanmedian(xvel)
yvel = yvel - np.nanmedian(yvel)
```

iii. The notes cite `findVelocity.m` and say paw velocity should be derived from bottom-camera tracking and split at the session median.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI thresholds paw velocity at the session 50th percentile over all valid paw-speed samples, producing binary `0/1` labels only.

ii.
```python
paw_thresh = np.nanpercentile(paw_speed_valid, 50)
paw_disc = discretize_time_series(paw_speed_valid, paw_thresh)
```

iii. The notes say continuous outputs were discretized with per-session 50th-percentile thresholds.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity is aligned the same way as tongue velocity: frame times are shifted by a session video offset and per-trial go cue, then interpolated to the neural time bins.

ii.
```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
...
interpolated = np.interp(time_axis, aligned_ft, spd)
speed[:, trial_idx] = interpolated.astype(np.float32)
```

iii. The notes describe a shared video-to-neural alignment procedure for video-derived outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from a separate `motionEnergy_<animal>_<date>.mat` file in the same session directory. The loader handles multiple nested MATLAB layouts and returns one trace per trial.

ii.
```python
def load_motion_energy(dirpath, animal, date):
    me_file = os.path.join(dirpath, f'motionEnergy_{animal}_{date}.mat')
    ...
    if hasattr(me_raw, 'dtype') and me_raw.dtype.names and 'data' in me_raw.dtype.names:
        me_data = me_raw['data'][0, 0]
        ...
    elif me_raw.dtype == np.dtype('O'):
        me_data = me_raw
```

iii. The notes explicitly mention multiple motion-energy file formats and separate loading from the `motionEnergy_*.mat` files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI interpolates each trial’s motion-energy trace onto the neural time axis using side-camera frame times or synthesized 400 Hz frame times, fills NaNs with nearest neighbors or zeros, and then sessionwise median-splits the result.

ii.
```python
me_interp[:, trial_idx] = np.interp(
    time_axis, aligned_ft[:n_frames], me_trial[:n_frames]
).astype(np.float32)
...
elif np.all(nan_mask):
    col[:] = 0.0
```

iii. The notes justify this as matching the reference idea of aligning motion energy to the neural time axis at the video frame rate.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI thresholds motion energy at the 50th percentile of all valid samples within a session and stores a binary 0/1 time series.

ii.
```python
me_thresh_50 = np.nanpercentile(me_valid, 50)
me_disc = discretize_time_series(me_valid, me_thresh_50)
```

iii. The notes say all three continuous outputs use a per-session median split.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy using video frame times from the side camera, subtracting session offset and per-trial go cue, then interpolating onto the neural time grid.

ii.
```python
ts, frame_times, _ = get_traj_data(data, fmt, 0, trial_idx)
...
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
...
me_interp[:, trial_idx] = np.interp(
    time_axis, aligned_ft[:n_frames], me_trial[:n_frames]
).astype(np.float32)
```

iii. The notes describe motion energy as a video-derived time series aligned to go cue using the same synchronization scheme as the other video features.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI generally fills or defaults missing values instead of preserving them as a separate state. Missing `stim.enable` becomes all-false; bad go-cue times are dropped; missing frame times are replaced with synthetic 400 Hz time stamps; missing tongue NaNs become zero velocity; remaining NaNs are nearest-filled; missing motion-energy files become all-zero traces; and late trials after recording end are excluded.

ii.
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
...
xvel[np.isnan(xvel)] = 0
yvel[np.isnan(yvel)] = 0
```
```python
if me_data is not None:
    ...
else:
    me_valid = np.zeros((len(time_axis), len(valid_trial_indices)), dtype=np.float32)
```

iii. The notes justify these fixes as practical edge-case handling needed to keep the decoder pipeline valid and to remove the two sessions’ late zero-neural trials.

## 11-a. What are the most time-consuming steps of the code?

i. The AI instruments and reports spike binning, tongue-velocity extraction, paw-velocity extraction, and motion-energy alignment as the main timed stages within each session. Its notes estimate roughly 7 to 9 seconds per session overall.

ii.
```python
t1 = time.time()
trialdat = bin_and_smooth_spikes(all_clusters, ntrials, align_times, time_axis, edges)
print(f"    Spike binning: {time.time()-t1:.1f}s")
...
t2 = time.time()
tongue_speed = extract_velocity_from_traj(...)
print(f"    Tongue velocity: {time.time()-t2:.1f}s")
...
t3 = time.time()
paw_speed = extract_velocity_from_traj(...)
...
t4 = time.time()
me_data, me_thresh = load_motion_energy(dirpath, animal, date)
```

iii. In `CONVERSION_NOTES.md`, the AI says no further speedups were needed because the full run was already about 5 to 7 minutes.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI leaves explicit loops over clusters, trials, and video-derived streams. Its notes specifically call out spike binning loops as vectorizable and say DLC trial loops are harder to remove because frame counts vary by trial.

ii.
```python
for i, clu in enumerate(clusters):
    ...
    for j in range(ntrials):
        ...
        counts = np.histogram(aligned_times, bins=edges)[0]
```
```python
for trial_idx in range(ntrials):
    try:
        ts, frame_times, feat_names = get_traj_data(data, fmt, view, trial_idx)
    except Exception:
        continue
```

iii. The notes say “Spike binning loops over neurons and trials (vectorized with `np.histogram`)” and “DLC extraction loops over trials (necessary due to variable frame times).”

## 11-c. What processing does the code repeat multiple times?

i. The AI repeats several small pieces of work: it rebuilds the same `input_data` array inside the per-trial output loop, fills NaNs in velocity/motion-energy traces in more than one pass, and computes some filtered variables that are never used later.

ii.
```python
for t_idx in range(len(valid_trial_indices)):
    input_data = time_axis.astype(np.float32).reshape(1, -1)
    input_trials.append(input_data)
```
```python
nan_cols = np.all(np.isnan(speed), axis=0)
speed[:, nan_cols] = 0.0
...
for t in range(ntrials):
    col = speed[:, t]
    nan_mask = np.isnan(col)
    ...
```

iii. No explicit justification for these repeated steps appears in the notes; they seem incidental to the implementation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes or stores several values that are not used downstream: `L_valid`, `miss_valid`, `align_times_valid`, `me_thresh`, and `keep_mask`; it also loads both `R` and `L` even though lick direction ultimately uses only `R`. Optional plotting support also preserves extra intermediate arrays solely for visualization.

ii.
```python
miss_valid = miss[valid_trial_indices]
R_valid = R[valid_trial_indices]
L_valid = L[valid_trial_indices]
autowater_valid = autowater[valid_trial_indices]
align_times_valid = align_times[valid_trial_indices]
```
```python
trialdat, filtered_clusters, keep_mask = remove_low_fr_neurons(trialdat, all_clusters)
...
me_data, me_thresh = load_motion_energy(dirpath, animal, date)
```

iii. The notes do not offer a specific justification for these unused values; they appear to be leftovers from intermediate exploration and diagnostics.
