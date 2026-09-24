# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 44 author-selected ephys sessions (25 fixed-delay and 19 randomized-delay), their folders, animals, dates, and ALM probes. It opens each `data_structure_*` file as HDF5 v7.3 or MATLAB v5 and loads the matching standalone motion-energy file separately.

ii.
```python
EPHYS_SESSIONS = [
    ('data/Ephys_Behavior', 'JEB6', '2021-04-18', [2]),
    ...
    ('data/RandomizedDelay_Ephys_Behavior', 'JEB24', '2023-11-03', [1]),
]
for sess_idx, (dirpath, animal, date, probes) in enumerate(sessions):
    result = process_session(dirpath, animal, date, probes, time_axis, edges, ...)

def load_mat_file(filepath):
    try:
        f = h5py.File(filepath, 'r')
        return f, 'h5'
    except Exception:
        return scipy.io.loadmat(filepath, squeeze_me=False), 'v5'
```

iii. The notes say the list and probes came from the authors' loader scripts; this intentionally excludes three randomized-delay files not used by those scripts and behavior-only datasets without neural data.

## 1-b. How are the data split into subjects?

i. The animal field in each hard-coded session tuple defines the subject. Unique animal names are sorted, and each retained session gets an index into that list.

ii.
```python
all_animals.append(animal)
unique_subjects = sorted(set(all_animals))
subject_idx = np.array([unique_subjects.index(a) for a in all_animals])
```

iii. The agent reports 14 animals and chose filenames/loader metadata because the session files are organized by animal and date.

## 1-c. How are the data split into sessions?

i. Each hard-coded `(directory, animal, date, probes)` tuple and corresponding `data_structure_<animal>_<date>.mat` is one session and one element of each top-level session list.

ii.
```python
session_id = f"{animal}_{date}"
filepath = os.path.join(dirpath, f"data_structure_{session_id}.mat")
all_neural.append(result['neural'])
all_input.append(result['input'])
all_output.append(result['output'])
```

iii. This follows the authors' per-session files and ALM-video loaders and yields the expected 44 sessions.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` sets the trial count. Behavioral vectors and go-cue times are indexed by zero-based trial, while spike `trial` labels are compared with `j + 1` because they are one-based. Each retained index produces one neural, input, and output array.

ii.
```python
ntrials = get_ntrials(data, fmt)
for j in range(ntrials):
    trial_num = j + 1
    spike_mask = trial == trial_num
...
for t_idx in range(len(valid_trial_indices)):
    neural_trials.append(trialdat_valid[:, :, t_idx].T.astype(np.float32))
```

iii. The agent notes that the Bpod fields, trajectories, and spike trial labels already supply trial membership, so boundaries need not be inferred.

## 1-e. How are trials filtered based on quality controls?

i. Trials with stimulation, early licking, absent/nonpositive go cues, or indices at/after the last trial containing a spike are removed. Sessions with fewer than two retained trials or fewer than ten units are skipped.

ii.
```python
valid_trials = ~stim_enable & ~early
valid_trials &= ~np.isnan(align_times) & (align_times > 0)
valid_trial_indices = np.where(valid_trials)[0]
max_spike_trial = max(int(np.max(c['trial'])) for c in all_clusters if len(c['trial']) > 0)
valid_trial_indices = valid_trial_indices[valid_trial_indices < max_spike_trial]
```

iii. Stimulated and early-lick trials are excluded by the paper. The last-spike cutoff was added after validation found 61 late behavioral trials with no ephys recording; the ten-unit rule comes from the paper.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the selected probes' `obj.clu` cluster fields `trialtm`, `trial`, and `quality`, plus `obj.bp.ev.goCue` for alignment.

ii.
```python
trialtm_ref = probe_data['trialtm'][i, 0]
trialtm = f[trialtm_ref][:].flatten()
trial_ref = probe_data['trial'][i, 0]
trial = f[trial_ref][:].flatten().astype(int)
align_times = get_event_times(data, fmt, ALIGN_EVENT)
```

iii. The notes identify these as the reference `alignSpikes`/`getSeq` inputs and concatenate the author-selected probe(s).

## 2-b. How is the `neural` data processed?

i. Per cluster and trial, spike times are shifted by that trial's go cue, histogrammed into 10 ms bins, divided by 0.01 s to make Hz, and passed through the agent's causal half-Gaussian length-15 convolution with reflected padding. Selected probes are concatenated; there is no normalization or baseline subtraction.

ii.
```python
aligned_times = trialtm[spike_mask] - align_times[j]
counts = np.histogram(aligned_times, bins=edges)[0]
rate = counts.astype(np.float64) / DT
smoothed = causal_gaussian_smooth(rate, SMOOTH_N, SMOOTH_BC)
```

iii. The agent chose the 10 ms tutorial setting and described the causal length-15 Gaussian and reflected boundary as matching `getSeq.m`/`mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters labeled (case-insensitively) `garbage`, `gabrga`, `noisy`, or `real?` are excluded. Units whose mean stored rate over all trials and bins is not greater than 1 Hz are then removed; sessions below ten remaining units are skipped.

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
if quality in EXCLUDE_QUALITIES:
    continue
mean_frs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
keep = mean_frs > LOW_FR_THRESHOLD
```

iii. The labels follow `findClusters.m`; 1 Hz and ten units follow the paper. Unlike the human solution, the agent did not additionally exclude `poor`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's within-trial time is shifted by its trial's `bp.ev.goCue`, then binned on a common window from -2.5 to +2.5 seconds.

ii.
```python
aligned_times = trialtm[spike_mask] - align_times[j]
counts = np.histogram(aligned_times, bins=edges)[0]
```

iii. The notes explicitly connect this subtraction to the authors' `alignSpikes.m` and the requested go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 500 nonoverlapping 10 ms bins over [-2.5, 2.5] s. Raw spikes are binned directly at 10 ms; video-frame values are interpolated to the bin centers.

ii.
```python
TMIN, TMAX = -2.5, 2.5
DT = 1.0 / 100
edges = np.arange(TMIN, TMAX + DT/2, DT)
time_axis = edges[:-1] + DT / 2
```

iii. The agent acknowledged both 10 ms and 5 ms reference settings, selecting 10 ms from `WorkingWithDataObjs.m`; the human reference selected the authors' 5 ms default.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a constructed common bin-center axis defined by `TMIN`, `TMAX`, and `DT`, conceptually relative to each trial's raw `bp.ev.goCue`.

ii.
```python
time_axis, edges = compute_time_axis()
input_data = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes say time from the requested alignment event is the sole decoder input and is identical across trials.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Edges spaced by 10 ms are created from -2.5 to +2.5 s; half a bin is added to obtain 500 bin centers, which are reshaped to `(1, 500)` and copied into every trial.

ii.
```python
edges = np.arange(TMIN, TMAX + DT/2, DT)
time_axis = edges[:-1] + DT / 2
```

iii. The agent says this follows the `getSeq.m` center-of-bin convention.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the centers of the exact edges used to histogram go-cue-shifted spikes, so each input column corresponds to the same neural bin.

ii.
```python
counts = np.histogram(aligned_times, bins=edges)[0]
input_data = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The shared axis was intentionally used for neural data and all time-varying variables.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Although the code loads `hit`, `miss`, `R`, and `L`, the saved label is derived only from `bp.R` for retained trials.

ii.
```python
hit = get_bp_field(data, fmt, 'hit').astype(bool)
miss = get_bp_field(data, fmt, 'miss').astype(bool)
R = get_bp_field(data, fmt, 'R').astype(bool)
L = get_bp_field(data, fmt, 'L').astype(bool)
lick_direction = R_valid.astype(np.float32)
```

iii. The notes recognized that `R/L` is instructed/correct side and initially derived actual lick from outcome, but ultimately chose the simpler trial-side label.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. `R=True` becomes right (1) and all other retained trials become left (0); the scalar is repeated across time. Misses are not reversed and ignored trials have no `none` category.

ii.
```python
lick_direction = R_valid.astype(np.float32)
out[0, :] = int(lick_direction[t_idx])
```

iii. The agent justified this as using instruction/stimulus direction while letting the outcome field encode correctness, despite the requested variable being actual lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived from the retained trials' `obj.bp.autowater` flag.

ii.
```python
autowater = get_bp_field(data, fmt, 'autowater').astype(bool)
autowater_valid = autowater[valid_trial_indices]
```

iii. The agent identified autowater as the direct marker for water-cued trials.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater trials are WC (0), otherwise DR (1), and this per-trial scalar is repeated across bins.

ii.
```python
behavioral_context = (~autowater_valid).astype(np.float32)
out[1, :] = int(behavioral_context[t_idx])
```

iii. This mapping follows the prompt and reference.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The code reads `bp.hit` and `bp.miss`, but the saved outcome uses only `hit`.

ii.
```python
hit = get_bp_field(data, fmt, 'hit').astype(bool)
miss = get_bp_field(data, fmt, 'miss').astype(bool)
outcome = hit_valid.astype(np.float32)
```

iii. The agent's mapping plan treated hit as correct and everything else as incorrect, omitting the requested ignore class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hits become correct (1); misses and nonresponses both become incorrect (0). The binary scalar is repeated over time.

ii.
```python
outcome = hit_valid.astype(np.float32)
out[2, :] = int(outcome[t_idx])
```

iii. The notes justify a binary correct/incorrect output but do not justify collapsing ignore, which the instructions explicitly list separately.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only side-camera `obj.traj` feature `tongue`: its x/y positions, `frameTimes`, and feature name. The code reads but does not use tracking likelihood. `bp.ev.goCue` and `sglx`/Bpod bit starts supply temporal alignment.

ii.
```python
tongue_speed = extract_velocity_from_traj(
    data, fmt, view=0, feat_name='tongue', ...)
xpos = ts[feat_idx, 0, :]
ypos = ts[feat_idx, 1, :]
```

iii. The agent considered a jaw proxy but chose the actual side-view tongue. It did not combine the bottom-camera `top_tongue`, unlike the human reference.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. X/y are left unsmoothed, differentiated by sample index, combined as Euclidean speed, NaN velocities are replaced with zero, and the trace is linearly interpolated to the common axis. Remaining missing samples are nearest-filled and entirely missing trials become zero.

ii.
```python
xvel = np.gradient(xpos_smooth)
yvel = np.gradient(ypos_smooth)
xvel[np.isnan(xvel)] = 0
yvel[np.isnan(yvel)] = 0
spd = np.sqrt(xvel**2 + yvel**2)
interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. The notes cite `findVelocity.m`, treat NaN tongue coordinates as not visible, and intentionally turn them into zero; they report the resulting strong low-class imbalance.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session-wide median across all retained trials/bins is used. Values below it are 0 and values at/above it are 1; a zero median is replaced by `1e-10`. There is no class 2 for not visible.

ii.
```python
tongue_thresh = np.nanpercentile(tongue_speed_valid, 50)
if threshold < 1e-10:
    threshold = 1e-10
return (values >= threshold).astype(np.float32)
```

iii. The median split follows the prompt, but the agent used the epsilon workaround to separate exact zero from movement rather than preserving invisibility as the required third class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The agent subtracts a session video offset and trial go cue from side-camera frame times, then uses `np.interp` at the 10 ms neural bin centers. The offset uses medians where available and falls back to 0.5 s.

ii.
```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
interpolated = np.interp(time_axis, aligned_ft, spd)
```

iii. This was intended to match `findVideoOffset.m`, but the reference takes modes and averages frames within bins rather than point-interpolating.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera `obj.traj` feature `top_paw`, specifically x/y and frame times; likelihood is not used. Go cues and bitcode timing provide alignment.

ii.
```python
paw_speed = extract_velocity_from_traj(
    data, fmt, view=1, feat_name='top_paw', ...)
```

iii. The agent selected `top_paw` as the reliable bottom-camera paw, matching the reference feature choice.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. X/y are causal-smoothed with window 21, differentiated by sample index, median velocity is subtracted per trial/axis, and magnitude is computed. Gaps are nearest-filled, full missing trials become zero, and values are linearly interpolated to bin centers.

ii.
```python
xpos_smooth = causal_gaussian_smooth(xpos, 21, 'reflect')
xvel = np.gradient(xpos_smooth)
xvel = xvel - np.nanmedian(xvel)
spd = np.sqrt(xvel**2 + yvel**2)
```

iii. The agent says smoothing and gradient follow the kinematics functions, but its procedure differs from the reference's likelihood-filtered contiguous-run, real-time derivative.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A median over every retained paw value in the session defines binary low/high classes, with the same epsilon adjustment for a degenerate zero threshold. Missing visibility is not a third class.

ii.
```python
paw_thresh = np.nanpercentile(paw_speed_valid, 50)
paw_disc = discretize_time_series(paw_speed_valid, paw_thresh)
```

iii. The session median is required, but missing data was filled first, so the specified `not visible` category cannot appear.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are offset-corrected and go-cue-shifted, then paw speed is point-interpolated at the 10 ms neural bin centers.

ii.
```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
speed[:, trial_idx] = np.interp(time_axis, aligned_ft, spd)
```

iii. The common clock/axis rationale is sound, but the reference uses mode-based offset and within-bin averaging.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It comes from the per-trial arrays in the standalone `motionEnergy_<animal>_<date>.mat`; side-camera frame times, go cues, and video offset fields align it.

ii.
```python
me_file = os.path.join(dirpath, f'motionEnergy_{animal}_{date}.mat')
me_raw = me_mat['me']
ts, frame_times, _ = get_traj_data(data, fmt, 0, trial_idx)
```

iii. The loader handles bare, singly wrapped, and doubly wrapped layouts, matching the heterogeneous files described in the notes.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The already-computed per-frame scalar is linearly interpolated to common bin centers; gaps are nearest-filled and wholly missing trials/files become zero. It is then median-split.

ii.
```python
me_interp[:, trial_idx] = np.interp(
    time_axis, aligned_ft[:n_frames], me_trial[:n_frames]
)
...
me_valid = np.zeros((len(time_axis), len(valid_trial_indices)), dtype=np.float32)
```

iii. The agent correctly avoided recomputing spatial motion energy, but chose interpolation and imputation rather than bin averaging and a no-video class.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A session-wide median over the filled/interpolated trace creates binary low/high values, with epsilon substitution when the median is effectively zero. There is no class 2 for no video.

ii.
```python
me_thresh_50 = np.nanpercentile(me_valid, 50)
me_disc = discretize_time_series(me_valid, me_thresh_50)
```

iii. The percentile matches the prompt, but the agent's output metadata lists only `['low', 'high']`, contrary to the required `no video` class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. It takes side-camera frame times, subtracts the video offset and go cue, truncates the trace and time array to their common length, then interpolates at neural bin centers.

ii.
```python
aligned_ft = frame_times - ft_offset - align_times[trial_idx]
n_frames = min(len(me_trial), len(aligned_ft))
me_interp[:, trial_idx] = np.interp(time_axis, aligned_ft[:n_frames], me_trial[:n_frames])
```

iii. The agent intended all streams to share one clock, but differs from the human solution's mode offset and per-bin averaging.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing frame times get synthetic 400 Hz times and a 0.5 s offset; partial trajectory gaps are nearest-filled; missing tongue values become zero; entirely missing video/motion-energy trials or files become zero. Missing/nonpositive go cues are dropped. Late behavioral trials after ephys ends are dropped, and differing camera lengths are processed separately.

ii.
```python
frame_times = np.arange(n_frames) / 400.0
ft_offset = 0.5
speed[:, nan_cols] = 0.0
...
valid_trials &= ~np.isnan(align_times) & (align_times > 0)
```

iii. The agent favored a complete NaN-free binary output and documented the late-recording fix. This conflicts with the explicit third missingness classes and fabricates values where the reference preserves missingness.

## 11-a. What are the most time-consuming steps of the code?

i. The agent times spike binning, tongue extraction, paw extraction, and motion energy per session; full conversion took 269 s (about 6.1 s/session). The nested spike neuron-by-trial histogram loop and per-trial video extraction/interpolation are the principal compute-heavy stages, in addition to file I/O.

ii.
```python
for i, clu in enumerate(clusters):
    for j in range(ntrials):
        ... np.histogram(aligned_times, bins=edges)
for trial_idx in range(ntrials):
    ... get_traj_data(...)
```

iii. The notes estimate 7–9 s/session and explicitly identify spike binning and DLC trial loops as inefficiencies, although they do not provide an aggregate timing breakdown.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike binning loops over every cluster and every trial, though trial and aligned time can be counted jointly with one 2-D histogram per cluster. NaN nearest-fill loops search every missing sample and could be vectorized/interpolated once. Output assembly and subject indexing also admit simple vectorization, while ragged trajectory reading reasonably remains per trial.

ii.
```python
for i, clu in enumerate(clusters):
    for j in range(ntrials):
        spike_mask = trial == j + 1
...
for idx in np.where(nan_mask)[0]:
    nearest = valid[np.argmin(np.abs(valid - idx))]
```

iii. The notes claim spike histograms were vectorized, but the final code only vectorizes within each single trial; the human reference's `histogram2d` demonstrates the missed vectorization.

## 11-c. What processing does the code repeat multiple times?

i. Every trajectory call rereads/dereferences `featNames`; tongue and paw each traverse all trials independently; motion energy traverses them a third time and rereads side-camera frame times. The same `time_axis.astype(...).reshape(...)` is created for every trial, and nearest filling is repeated per trace.

ii.
```python
for trial_idx in range(ntrials):
    ts, frame_times, feat_names = get_traj_data(...)
...
input_data = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The agent notes only that DLC loops are necessary for ragged frames; it does not identify the repeated dereferencing, passes, or input conversion.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `L` but labels direction only from `R`, loads `miss` but labels outcome only from `hit`, loads motion-energy `moveThresh` but replaces it with a median, and retains cluster dictionaries after only their count is needed. Optional diagnostic plotting also computes/plots continuous signals that are not saved, though only when requested.

ii.
```python
miss = get_bp_field(data, fmt, 'miss').astype(bool)
L = get_bp_field(data, fmt, 'L').astype(bool)
me_data, me_thresh = load_motion_energy(dirpath, animal, date)
me_thresh_50 = np.nanpercentile(me_valid, 50)
```

iii. The unused behavior fields reflect an abandoned plan to infer actual lick/outcome correctly. Ignoring `moveThresh` is appropriate because the prompt specifically asks for a session 50th percentile; the plotting path is justified as optional validation.
