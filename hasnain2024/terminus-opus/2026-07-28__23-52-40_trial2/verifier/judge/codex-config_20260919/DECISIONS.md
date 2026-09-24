# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes 25 sessions from `data/Ephys_Behavior` in `SESSION_META`, constructs one `data_structure_*.mat` and one `motionEnergy_*.mat` path per session, and reads the former directly with `h5py`. It does not load the randomized-delay directory.

ii.
```python
DATA_DIR = 'data/Ephys_Behavior'
SESSION_META = [('EKH1', '2021-08-07', [1]), ...]
for idx, (animal, date, probes) in enumerate(sessions):
    result = process_session(animal, date, probes, ...)
```

iii. The notes say this folder was chosen because it contains the DR+WC two-context task and report 25 sessions as matching the paper. The AI treated that paper subset as the complete target dataset.

## 1-b. How are the data split into subjects?

i. Subject identity is the hard-coded `animal` string. Unique animals among successfully processed sessions are sorted, and each session receives its index in that list.

ii.
```python
subjects = sorted(list(set(r['animal'] for r in all_results)))
subj_idx = subjects.index(result['animal'])
subject_idx_list.append(subj_idx)
```

iii. The notes justify using all 10 animals found in the selected files despite the paper reporting nine.

## 1-c. How are the data split into sessions?

i. Every `(animal, date, probes)` tuple is one session and becomes one list element, unless processing returns `None` because the file, trials, clusters, or minimum neuron count fail checks.

ii.
```python
session_id = f'{animal}_{date}'
data_file = os.path.join(DATA_DIR, f'data_structure_{session_id}.mat')
if result is not None:
    all_results.append(result)
```

iii. The AI says the tuples were transcribed from the authors' loading scripts and that 25 sessions matches the fixed-delay paper analysis.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines trial count. Spike `trial` labels select spikes for each trial; behavioral arrays and camera cells are indexed by the same zero-based trial index. Only indices in `valid_trials` are finally packaged.

ii.
```python
Ntrials = int(bp['Ntrials'][0, 0])
valid_trials = np.where(valid_mask)[0]
for trial_idx in valid_trials:
    neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32))
```

iii. The rationale is that the MATLAB objects already provide trial assignments and per-trial arrays, so boundaries need not be inferred.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained only if they are non-early, non-stimulation, non-ignore, and explicitly hit or miss. Sessions with fewer than two retained trials are dropped. There is no cutoff for behavioral trials recorded after electrophysiology ended.

ii.
```python
valid_mask = (early == 0) & (no == 0) & (stim_enable == 0) & ((hit == 1) | (miss == 1))
if len(valid_trials) < 2:
    return None
```

iii. The notes state that the paper excludes early, no-response, and stimulation trials and therefore retain only hits and misses.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each selected probe's `obj.clu` entries: `quality`, `trialtm`, and `trial`, plus `bp.ev.goCue` for alignment. `tm` is read but never used.

ii.
```python
spike_trialtm = f[tm_ref][:].flatten()
spike_trial = f[trial_ref][:].flatten().astype(int)
spike_tm_abs = f[tm_abs_ref][:].flatten()
spike_goCue = goCue[spike_trial_valid - 1]
```

iii. The AI cites the reference `alignSpikes.m`: subtract each trial's go cue from trial-relative spike time.

## 2-b. How is the `neural` data processed?

i. For every cluster and trial, aligned spikes are histogrammed into 10 ms bins, divided by 0.01 to obtain Hz, and passed through a 15-sample causal Gaussian smoother with reflected prefix padding. Selected probes are concatenated.

ii.
```python
counts, _ = np.histogram(trial_spikes, bins=EDGES)
fr = counts / PARAMS['dt']
fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
trialdat = np.concatenate(all_trialdat, axis=1)
```

iii. The notes claim this matches `getSeq.m` and `mySmooth.m`, choosing the causal 15-sample kernel used in the scripts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Quality strings equal to `garbage`, misspelled `gabrga`, `noisy`, or `real?` are excluded case-insensitively; empty labels remain. Neurons with mean smoothed rate at or below 1 Hz are removed, and sessions with fewer than 10 remaining neurons are dropped. `poor` units are not excluded.

ii.
```python
'quality_exclude': ['garbage', 'gabrga', 'noisy', 'real?']
meanFRs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
fr_mask = meanFRs > PARAMS['lowFR']
if n_neurons < 10:
    return None
```

iii. The AI says the label list matches MATLAB `findClusters.m`, while 1 Hz and 10 neurons follow the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's trial-relative time is shifted by that trial's go-cue time before binning from -2.5 to +2.5 seconds.

ii.
```python
spike_goCue = goCue[spike_trial_valid - 1]
spike_aligned = spike_trialtm_valid - spike_goCue
```

iii. The AI explicitly identifies this as the operation in the authors' `alignSpikes.m` with `alignEvent='goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI creates 500 non-overlapping 10 ms bins across five seconds. Raw spikes are rebinned by histogramming; camera variables are interpolated onto the same bin-center axis.

ii.
```python
'dt': 1/100
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
```

iii. The notes acknowledge a conflicting 5 ms default but choose 10 ms because it appeared in “most scripts” and was viewed as the standard analysis resolution.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is generated from the configured -2.5-to-2.5 s bin edges rather than read from a raw variable; `goCue` defines the zero used when aligning spikes and video.

ii.
```python
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
```

iii. The AI describes the shared axis as a direct continuous representation of time relative to go cue.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. It takes the centers of uniform 10 ms bins and copies the same `(1, 500)` array into every trial.

ii.
```python
EDGES = np.arange(-2.5, 2.5 + 0.01, 0.01)
TIME_AXIS = EDGES[:-1] + 0.01 / 2
```

iii. No additional processing is justified because alignment makes this deterministic.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the centers of exactly the edges used to histogram go-cue-shifted spikes, so corresponding input and neural columns share a bin.

ii.
```python
counts, _ = np.histogram(trial_spikes, bins=EDGES)
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
```

iii. The AI's validation notes report the expected axis `[-2.495, 2.495]` and matching shapes.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived solely from `bp.R`; `bp.L` is read but unused. Because ignore trials are removed, every retained value is 0 for a left-instructed trial or 1 for a right-instructed trial.

ii.
```python
L = bp['L'][:].flatten()
R = bp['R'][:].flatten()
lick_direction = R.copy()
```

iii. The mapping plan equates instructed right/left with lick direction and expects approximately balanced classes.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. No behavioral-outcome correction is made: `R` is cast to an integer and repeated across all time bins. Thus miss trials are labeled by instructed side rather than the opposite port actually licked, and there is no `none` class.

ii.
```python
'lick_direction': int(lick_direction[trial_idx])
lick_dir = np.full(N_TIMEBINS, info['lick_direction'], dtype=np.int64)
```

iii. The AI believed the raw side flag directly represented lick direction and justified two classes because ignore trials had been filtered out.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived from the per-trial `bp.autowater` flag.

ii.
```python
autowater = bp['autowater'][:].flatten()
context = 1 - autowater
```

iii. The notes identify autowater trials as water-cued and all others as delayed-response.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The binary flag is inverted so WC/autowater is 0 and DR is 1, cast to integer, and repeated through time.

ii.
```python
context = 1 - autowater  # DR=1, WC=0
context = np.full(N_TIMEBINS, info['context'], dtype=np.int64)
```

iii. This follows the class order documented in `output_values`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived only from `bp.hit`; `miss` and `no` participate in filtering but not the final value.

ii.
```python
hit = bp['hit'][:].flatten()
miss = bp['miss'][:].flatten()
no = bp['no'][:].flatten()
outcome = hit.copy()
```

iii. Since retained trials are restricted to hits and misses, the AI treats hit as a sufficient binary correct/incorrect indicator.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit becomes correct (1), retained non-hit/miss becomes incorrect (0), and the value is repeated through time. Ignore trials and the required ignore class are absent.

ii.
```python
outcome = hit.copy()  # 1=correct, 0=incorrect
outcome = np.full(N_TIMEBINS, info['outcome'], dtype=np.int64)
```

iii. The notes explicitly choose to exclude no-response trials “per paper methods,” notwithstanding the decoder specification.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses side-view `obj.traj`: feature name `tongue`, per-trial `ts` x/y coordinates, `frameTimes`, and `NdroppedFrames`. Bitcode and go-cue fields provide clock alignment. It does not use likelihood or the bottom-view `top_tongue` track.

ii.
```python
traj_ref = f['obj']['traj'][0, 0]
if name == 'tongue': tongue_idx = i
x = ts_data[tongue_idx, 0, :]
y = ts_data[tongue_idx, 1, :]
```

iii. The notes choose the side-view tongue feature and claim this matches the reference kinematics functions.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. X and y positions are linearly interpolated to 10 ms bin centers, differentiated with `np.gradient` by sample index, NaN derivatives are replaced by zero, and Euclidean magnitude is taken. There is no likelihood filtering, positional smoothing, division by elapsed seconds, second-camera combination, or view normalization.

ii.
```python
xpos[:, trix] = fx(taxis)
xv = np.gradient(xpos[:, trix]); xv[np.isnan(xv)] = 0
yv = np.gradient(ypos[:, trix]); yv[np.isnan(yv)] = 0
speed = np.sqrt(xvel**2 + yvel**2)
```

iii. The AI says gradient-of-position with tongue NaNs set to zero matches `findVelocity.m`.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A session-wide median is computed across all retained trial/time values. If it is zero, the median of positive values replaces it. Values below the threshold become 0 and values greater than or equal become 1; no “not visible” class is produced.

ii.
```python
threshold = np.percentile(all_values, 50)
if threshold == 0:
    threshold = np.percentile(all_values[all_values > 0], 50)
disc = (v >= threshold).astype(np.float32)
```

iii. The positive-only fallback was introduced because missing tongue tracking had been encoded as zero and otherwise dominated the median.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. A session video offset is computed from modal bitcode times. Each frame time is shifted by that offset and its trial's go cue, then position is interpolated onto `TIME_AXIS`.

ii.
```python
vidshift = vidFileOffset - bitStart
aligned_times = frameTimes - vidshift - goCue[trix]
xpos[:, trix] = interp1d(aligned_times[valid], x[valid], ...)(taxis)
```

iii. The AI cites `findVideoOffset.m` and uses the same target axis as neural data.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses bottom/top-view `obj.traj` fields and selects every feature whose name contains `paw` (normally `top_paw` and `bottom_paw`), plus frame, bitcode, and go-cue timing.

ii.
```python
traj_ref = f['obj']['traj'][1, 0]
if 'paw' in name.lower():
    paw_indices.append(i)
```

iii. The notes state that top-view top and bottom paw speeds are averaged.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw, positions are interpolated to the target axis and extrapolated gaps are nearest-filled. Gradients are computed, their median first differences are subtracted, missing gradients are nearest-filled, speed magnitude is calculated, and speeds are averaged across paw features.

ii.
```python
xp = np.interp(indices, indices[mask], xp[mask])
xv = np.gradient(xpos[:, trix])
xv = xv - np.nanmedian(np.diff(xpos[:, trix]))
speed = np.sqrt(xvel**2 + yvel**2)
avg_speed = np.mean(all_speeds, axis=0)
```

iii. The AI says this follows non-tongue behavior in `findVelocity.m` and combines both paw features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The same per-session median/positive-median fallback creates only low (0) and high (1). There is no “not visible” category because gaps are filled or default to zeros.

ii.
```python
paw_disc = discretize_continuous(result['paw_vel'])
disc = (v >= threshold).astype(np.float32)
```

iii. The AI applies the prompt's 50th percentile but does not discuss preserving missing visibility as class 2.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-view frame times are corrected by the bitcode-derived video offset and trial go cue, then positions are interpolated to the neural bin centers.

ii.
```python
aligned_times = frameTimes - vidshift - goCue[trix]
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
xp = fx(taxis)
```

iii. The AI uses the common time grid and reports this as matching the reference interpolation.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It loads `me.data` from the standalone session motion-energy MAT file, unwrapping one nested struct when present. Side-camera `frameTimes`, bitcode timing, and go cue supply alignment.

ii.
```python
me_raw = sio.loadmat(me_file)
me_data = me_raw['me'][0, 0]['data']
if me_data.dtype.names is not None and 'data' in me_data.dtype.names:
    me_data = me_data[0, 0]['data']
```

iii. The notes say the nested layout was discovered in JEB15 and handled like the authors' MATLAB loader.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Each trace is length-matched to side-camera timestamps, linearly interpolated at 10 ms centers, and out-of-range NaNs are nearest-filled. Failed trials retain an initialized all-zero trace.

ii.
```python
me_interp = interp1d(aligned_times[valid], me_trial[valid],
                     bounds_error=False, fill_value=np.nan)(taxis)
me_interp = np.interp(indices, indices[mask], me_interp[mask])
```

iii. The AI describes interpolation as matching `loadMotionEnergy.m` and considers all-zero sessions a property of the data.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. It is split using the per-session median with the same positive-only fallback when the median is zero. Only low/high classes are emitted; no “no video” class exists.

ii.
```python
me_disc = discretize_continuous(result['motion_energy'])
disc = (v >= threshold).astype(np.float32)
```

iii. The AI follows the 50th-percentile instruction but treats absent values as zero/fill rather than a distinct category.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-view camera frame times are corrected by the session video offset and trial go cue, and the values are interpolated onto the neural bin-center axis.

ii.
```python
aligned_times = frameTimes - vidshift - goCue[trix]
me_interp = f_interp(taxis)
```

iii. The AI says motion energy is frame-based and should use the same video-clock correction as tracked movement.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Broad `try/except` blocks skip malformed trials or whole streams. Missing tongue samples become zero velocity; paw and motion gaps are nearest-filled; unavailable whole streams and failed trials become all-zero arrays. Synthetic 400 Hz timestamps are used if frame times cannot be read. Missing files or too-small sessions are skipped.

ii.
```python
except:
    frameTimes = np.arange(1, n_frames + 1) / 400.0
...
else:
    tongue_vel_trials.append(np.zeros(N_TIMEBINS, dtype=np.float32))
```

iii. The notes frame these branches as edge-case robustness, but also introduce a positive-median threshold to compensate for zeros used when tongue is invisible.

## 11-a. What are the most time-consuming steps of the code?

i. The AI's notes estimate roughly 6–7 seconds per session and about 180 seconds total, but do not profile individual functions. The nested cluster-by-trial spike histograms, repeated HDF5 dereferences, camera interpolation loops, and plotting when enabled are the evident expensive stages.

ii.
```python
for ci, clu_idx in enumerate(cluid):
    for trial_num in range(1, Ntrials + 1):
        counts, _ = np.histogram(trial_spikes, bins=EDGES)
```

iii. The documentation reports runtime estimates rather than a measured breakdown.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike counting loops over every cluster and then every trial, although trial and aligned-time labels could be histogrammed together. Smoothing loops over columns; camera processing repeats trial loops for each paw and again for gradients; final packaging also loops over trials.

ii.
```python
for trial_num in range(1, Ntrials + 1):
    trial_mask = spike_trial_valid == trial_num
...
for paw_idx in paw_indices:
    for trix in range(Ntrials):
```

iii. The AI did not document these vectorization opportunities; it described the implementation as matching MATLAB.

## 11-c. What processing does the code repeat multiple times?

i. `find_video_offset` is called independently for tongue, paw, and motion energy. Trajectory feature names and per-trial HDF5 trajectory data are reread for tongue and each paw. Paw position and derivative loops repeat per feature, and `TIME_AXIS` is copied for every trial.

ii.
```python
vidshift = find_video_offset(f)  # in all three stream functions
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
```

iii. The notes do not acknowledge this repetition and instead emphasize functional correspondence with separate reference routines.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads absolute spike times `tm` but never uses them, reads `L` without using it, computes an initial median in `find_video_offset` before overwriting it with the mode, and processes all trials/neurons before retaining only valid trials and low-rate neurons. Optional plotting recomputes discretization and produces diagnostics not stored in the dataset.

ii.
```python
spike_tm_abs = f[tm_abs_ref][:].flatten()
L = bp['L'][:].flatten()
bitStart = np.median(...)
bitStart = stats.mode(...).mode
```

iii. These discarded computations are not discussed in the notes; plotting is explicitly offered only for visual verification.
