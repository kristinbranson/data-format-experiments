# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 45 electrophysiology/video sessions, including their animal, date, probe(s), and one of two data folders. It opens each `data_structure_*.mat` through a custom `SessionData` reader supporting MATLAB v7.3/HDF5 and v5 files, and separately loads the matching motion-energy file. Missing session files are skipped.

ii.
```python
EPHYS_SESSIONS = [
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ...
    ('JEB24', '2023-11-03', [1], 'RandomizedDelay_Ephys_Behavior'),
]
sd = SessionData(fpath)
me_data, me_thresh = load_motion_energy(data_dir, anm, date)
```

iii. The trajectory says the list was transcribed from the authors' recording/video loading scripts and that both MAT formats and multiple motion-energy layouts had to be handled. It reported 45 sessions as “all” relevant sessions.

## 1-b. How are the data split into subjects?

i. Subject identity is the `anm` value stored in each hard-coded session tuple. Unique subjects are accumulated in first-occurrence order, and each retained session receives the corresponding index.

ii.
```python
if anm not in subjects_set:
    subjects_set.append(anm)
subject_idx = np.array([subjects.index(s['anm']) for s in all_sessions])
```

iii. The trajectory treats animal identifiers in the filenames/loading scripts as the reliable subject definition and reports 14 mice.

## 1-c. How are the data split into sessions?

i. Each `<animal>_<date>` data file is one session and becomes one outer-list element of `neural`, `input`, and `output`; sessions failing file, trial, cluster, or unit-count checks are skipped.

ii.
```python
for anm, date, probes, data_dir in available:
    result = process_session(anm, date, probes, data_dir, time_edges, time_axis)
    if result is not None:
        all_sessions.append(result)
```

iii. The agent justified this as matching the authors' session organization and validated each resulting session separately.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines the trial count. Trial-indexed behavioral arrays and go cues use row/index `j`; spike `clu.trial` values are matched to `j + 1`; video and motion-energy arrays use `trial_idx`. Each selected trial becomes one list entry.

ii.
```python
ntrials = sd.get_ntrials()
trial_mask = (trial_nums == (j + 1))
for trial_idx in valid_idx:
    neural_trials.append(trialdat[:, :, trial_idx])
```

iii. The trajectory recognized MATLAB spike trial numbers as 1-based and Python behavioral/video indexing as 0-based.

## 1-e. How are trials filtered based on quality controls?

i. It retains only hit or miss trials, excluding photostimulation, early-lick, and ignore/no-response trials. It skips sessions with fewer than five such trials initially and fewer than two output trials finally. It does not remove trials occurring after electrophysiology ended.

ii.
```python
valid_trials = (hit | miss) & ~stim_enable & ~early
valid_idx = np.where(valid_trials)[0]
if len(valid_idx) < 5:
    return None
```

iii. The trajectory says these conditions match examples in the reference analysis and explicitly chose to exclude ignore trials; later validation noted roughly 30 retained all-zero neural trials but did not remove them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from selected probes in `obj.clu`: each cluster's `quality`, 1-based spike `trial`, and within-trial `trialtm`, plus `bp.ev.goCue` for alignment.

ii.
```python
clusters = sd.get_clusters(probe_nums)
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    aligned_times = spike_times[trial_mask] - gocue[j]
```

iii. The agent identified these as the fields used by the authors' spike-alignment routines.

## 2-b. How is the `neural` data processed?

i. For every unit and trial, aligned spikes are histogrammed, divided by 0.01 s to form Hz, and passed through a length-15 causal half-Gaussian convolution with a reflected prefix. Probe populations are concatenated. No normalization or baseline subtraction is applied.

ii.
```python
counts, _ = np.histogram(aligned_times, bins=time_edges)
fr = counts / DT
trialdat[i, :, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. The trajectory claims this matches `mySmooth.m`/`getSeq.m`: MATLAB `gausswin`, its first half zeroed, and reflect boundary handling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters labeled `garbage`, misspelled `gabrga`, `noisy`, or `real?` are removed. Mean firing rate is computed over retained trials and all bins; units at or below 1 Hz are removed. Sessions with fewer than 10 remaining units are skipped. `poor` units remain.

ii.
```python
EXCLUDE_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
mean_frs = np.mean(np.mean(trialdat[:, :, valid_idx], axis=2), axis=1)
keep_units = mean_frs > LOW_FR_THRESH
```

iii. The agent cited `findClusters.m` with quality `all` and the paper's 1 Hz cutoff, and regarded the 10-unit session cutoff as needed for decoding.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's within-trial time has that trial's go-cue time subtracted before histogramming over −2.5 to +2.5 seconds.

ii.
```python
aligned_times = spike_times[trial_mask] - gocue[j]
counts, _ = np.histogram(aligned_times, bins=time_edges)
```

iii. The trajectory describes direct subtraction as matching the authors' go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent uses 10 ms non-overlapping bins (500 bin centers from −2.495 to +2.495 s). Raw spikes are binned at this resolution; video series are interpolated onto those centers.

ii.
```python
DT = 1/100  # 10 ms
time_edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = time_edges[:-1] + DT / 2
```

iii. The trajectory repeatedly states that 10 ms was selected to match its reading of `WorkingWithDataObjs.m`/`getSeq.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read per trial from a raw field. It is the agent-defined centers of the common −2.5 to +2.5 s go-cue-aligned bin grid.

ii.
```python
time_axis = time_edges[:-1] + DT / 2
input_data = time_axis.reshape(1, -1).copy()
```

iii. The agent treated this as the natural continuous coordinate after aligning all trials to `bp.ev.goCue`.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. It constructs bin edges with `np.arange`, converts them to centers by adding 5 ms, reshapes to `(1, 500)`, and copies the identical array for every trial.

ii.
```python
time_axis = time_edges[:-1] + DT / 2
input_data = time_axis.reshape(1, -1).copy()
```

iii. The trajectory gives no separate justification beyond consistency with the shared bin grid.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input values are the centers of the exact edges used to histogram the go-cue-relative spikes, so columns correspond one-to-one.

ii.
```python
counts, _ = np.histogram(aligned_times, bins=time_edges)
input_data = time_axis.reshape(1, -1).copy()
```

iii. The agent explicitly intended neural and all time-varying variables to share one axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It reads `bp.R` (and also loads `bp.L`, though `L` is unused). Direction is assigned directly from the instructed-right flag for every retained hit or miss trial; outcome is not used to reverse error-trial direction.

ii.
```python
R = sd.get_trial_array('R').astype(bool)
L = sd.get_trial_array('L').astype(bool)
lick_dir = 1 if R[trial_idx] else 0
```

iii. The trajectory describes this as “left=0, right=1 (per-trial, from bp.R)” and did not discuss the distinction between instruction and actual lick on misses.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. `R=True` maps to right/1 and otherwise left/0, then the scalar is repeated across all 500 time bins. Ignore trials and a no-lick category are absent.

ii.
```python
lick_dir = 1 if R[trial_idx] else 0
out[0, :] = t['lick_dir']
```

iii. The trajectory says this directly implements the requested binary codes, relying on its earlier decision to discard ignores.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived from the per-trial `bp.autowater` field.

ii.
```python
autowater = sd.get_trial_array('autowater')
```

iii. The agent understood autowater trials as water-cued and all others as delayed-response.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Exact value 1 maps to WC/0; anything else maps to DR/1. The scalar is repeated over time.

ii.
```python
context = 0 if autowater[trial_idx] == 1 else 1
out[1, :] = t['context']
```

iii. The trajectory says this follows the prompt's WC=0, DR=1 coding.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is derived from `bp.hit`; `bp.miss` is loaded and used for trial inclusion, while `bp.no` is not loaded.

ii.
```python
hit = sd.get_trial_array('hit').astype(bool)
miss = sd.get_trial_array('miss').astype(bool)
outcome = 1 if hit[trial_idx] else 0
```

iii. The agent assumed every retained non-hit is a miss because ignores were filtered out.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit maps to correct/1 and otherwise incorrect/0; the value is repeated across time. There is no ignore/2 class.

ii.
```python
outcome = 1 if hit[trial_idx] else 0
out[2, :] = t['outcome']
```

iii. The trajectory says this matches the requested binary outcome labels and its choice to exclude no-response trials.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the bottom-camera `top_tongue` feature (or the first bottom-camera feature containing “tongue” as fallback), specifically x, y, and frame times. Tracking likelihood is not returned or used. Video/behavior bitcodes, sampling rate, and go cue supply clock alignment.

ii.
```python
if name == 'top_tongue':
    tongue_feat_idx = idx
ft, x, y = sd.get_trial_traj(1, trial_idx, tongue_feat_idx)
```

iii. The trajectory says bottom-camera `top_tongue` was chosen as the tongue signal; it did not justify omitting the side-camera tongue view.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. It differences raw x and y, takes Euclidean displacement divided by a fixed 1/400 s, timestamps at assumed midframes, and linearly interpolates velocity to 10 ms neural-bin centers. It applies no confidence filter, coordinate smoothing, real-time derivative, view normalization, or two-view combination.

ii.
```python
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
vel_t = ft[:-1] + dt_vid / 2
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The trajectory characterized this as velocity magnitude at 400 Hz interpolated to neural bins and noted tongue visibility/skew, but did not revise the processing.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session-wide median is computed over all non-NaN interpolated values. Values below it are 0 and values at/above it are 1. NaNs are also forced to 0; no “not visible” class is represented.

ii.
```python
tongue_thresh = np.nanpercentile(all_tongue, 50)
tongue_disc = np.where(np.isnan(t['tongue_vel']), 0,
                       np.where(t['tongue_vel'] >= tongue_thresh, 1, 0))
```

iii. The median split follows the requested percentile. The trajectory explicitly says NaNs outside video coverage were assigned class 0, apparently to keep binary outputs.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video-clock offset is computed from bitcode modes. The offset and trial go cue are subtracted from velocity timestamps, which are then interpolated at the neural bin centers.

ii.
```python
vidshift = sd.get_video_offset()
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
tongue_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The trajectory cites `findVideoOffset.m` and intended the same clock correction for all video-derived signals.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom-camera `bottom_paw` (or the first feature containing “paw” as fallback), taking x, y, and frame times but not likelihood.

ii.
```python
if name == 'bottom_paw':
    paw_feat_idx = idx
ft, x, y = sd.get_trial_traj(1, trial_idx, paw_feat_idx)
```

iii. The trajectory identifies `bottom_paw` as its chosen paw feature but gives no comparison with `top_paw` tracking reliability.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Raw coordinate differences are converted to speed using fixed 400 Hz timing and linearly interpolated to 10 ms centers. There is no likelihood filtering, smoothing, or derivative using actual frame intervals.

ii.
```python
vel = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt_vid
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The trajectory says paw processing is the same as tongue processing.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A session-wide non-NaN median is used; below is 0 and at/above is 1. Missing/untracked values are also encoded as 0, not category 2.

ii.
```python
paw_thresh = np.nanpercentile(all_paw, 50)
paw_disc = np.where(np.isnan(t['paw_vel']), 0,
                    np.where(t['paw_vel'] >= paw_thresh, 1, 0))
```

iii. The trajectory cites the requested per-session 50th percentile and binary classification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera velocity midpoints are corrected by the session bitcode offset and trial go cue, then interpolated onto neural centers.

ii.
```python
vel_t_aligned = vel_t - vidshift - gocue[trial_idx]
paw_vel = np.interp(time_axis, vel_t_aligned, vel, left=np.nan, right=np.nan)
```

iii. The agent intended identical video/neural alignment for tongue and paw.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It reads one motion-energy trace per trial from a separate `motionEnergy_<animal>_<date>.mat`, supporting wrapped struct and bare-cell layouts. It also reads `moveThresh` when present but never uses it. Bottom-camera frame times are used for timing.

ii.
```python
me_data, me_thresh = load_motion_energy(data_dir, anm, date)
ft = sd.get_trial_frame_times(1, trial_idx)
```

iii. The trajectory says loaders were generalized because files have multiple layouts and describes the trace as already-computed framewise motion energy.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. No spatial or temporal motion-energy calculation is redone. The stored per-frame trace is linearly interpolated to 10 ms centers; the file's `moveThresh` is discarded.

ii.
```python
me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx],
                      left=np.nan, right=np.nan)
```

iii. The trajectory considered the source trace already processed and used the instructed median rather than the authors' stored movement threshold.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The session median across non-NaN interpolated samples defines 0 below and 1 at/above. Missing samples or an entirely missing file become 0; there is no “no video” category 2.

ii.
```python
me_thresh_50 = np.nanpercentile(all_me, 50) if not np.all(np.isnan(all_me)) else 0
me_disc = np.where(np.isnan(t['motion_energy']), 0,
                   np.where(t['motion_energy'] >= me_thresh_50, 1, 0))
```

iii. The agent followed the per-session 50th-percentile instruction but acknowledged that a session lacking motion energy became all class 0.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The agent subtracts the session video offset and trial go cue from bottom-camera frame times, then interpolates the trace onto neural bin centers.

ii.
```python
ft_aligned = ft - vidshift - gocue[trial_idx]
me_interp = np.interp(time_axis, ft_aligned, me_data[trial_idx], left=np.nan, right=np.nan)
```

iii. The trajectory says all video signals use the same bitcode-derived correction; it does not justify using bottom rather than side-camera times for motion energy.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing files can skip sessions; inaccessible clusters/probes can be skipped; feature lookup and per-trial video errors are broadly caught. Missing video samples remain NaN until discretization, when they are silently mapped to class 0. Motion-energy loader failures return `(None, None)`. No confidence-based missingness is detected, and late all-zero neural trials are retained.

ii.
```python
except:
    pass
...
tongue_disc = np.where(np.isnan(t['tongue_vel']), 0, ...)
```

iii. The trajectory emphasizes robustness to heterogeneous MAT layouts. It acknowledged all-zero late trials and unavailable motion energy but accepted them after validation rather than representing missingness explicitly.

## 11-a. What are the most time-consuming steps of the code?

i. The code does not instrument runtime, but its nested unit-by-trial spike histogram/smoothing loop, repeated video interpolation, file loading, accumulation of large float64 arrays, pickle writing, and subsequent decoder validation are the likely expensive stages.

ii.
```python
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        ...
        trialdat[i, :, j] = causal_gaussian_smooth(...)
```

iii. The trajectory focused on full conversion/decoder completion and did not provide profiling; its notes chiefly discuss the 3 GB output and validation runs.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner loop over every trial for every unit could be replaced by joint 2-D histogramming by trial and time. The convolution helper's column loop and the loops constructing thresholded outputs could also be vectorized. Ragged video trials still require some per-trial handling.

ii.
```python
for i, (q, trial_nums, spike_times) in enumerate(filtered_clusters):
    for j in range(ntrials):
        trial_mask = (trial_nums == (j + 1))
```

iii. The trajectory did not identify these optimization opportunities; it prioritized correctness and successful validation.

## 11-c. What processing does the code repeat multiple times?

i. For each unit it scans every trial and repeatedly builds masks; smoothing separately reconstructs the same Gaussian kernel for every nonempty unit-trial pair. Each video trial separately repeats feature access, alignment, and interpolation. The fixed time input is copied for every trial.

ii.
```python
kern = windows.gaussian(N, std=(N-1)/(2*2.5))
...
input_data = time_axis.reshape(1, -1).copy()
```

iii. The trajectory does not justify these repetitions; they arise from a straightforward implementation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `L` but uses only `R`, reads motion-energy `moveThresh` but recomputes a median, constructs `sample_data.pkl` although the requested full dataset is the product, and builds/saves extensive duplicated time/output rows. It also processes all trials into `trialdat` before retaining only `valid_idx` and computes rates for units later dropped.

ii.
```python
L = sd.get_trial_array('L').astype(bool)
me_data, me_thresh = load_motion_energy(...)
sample_data = {'neural': neural[:n_sample], ...}
```

iii. The trajectory required a sample file and validation artifacts under its original prompt, so sample creation was intentional there; `L` and `me_thresh` have no documented downstream use.
