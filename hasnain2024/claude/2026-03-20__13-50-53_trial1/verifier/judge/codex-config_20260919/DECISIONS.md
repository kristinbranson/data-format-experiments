# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the 25 fixed-delay and 19 randomized-delay sessions and their selected probes, then loads each `data_structure_<animal>_<date>.mat` plus its motion-energy file. It uses `mat73` for v7.3 files and a custom SciPy converter for v5 files.

ii.
```python
ALL_SESSIONS = EPHYS_SESSIONS + RANDOMIZED_DELAY_SESSIONS
obj = mat73.loadmat(data_path)['obj']
# fallback
obj = _load_v5_session(data_path)
me_file = scipy.io.loadmat(me_path)
```

iii. The notes say the registry was transcribed from the authors' loading scripts so commented-out/extra files are excluded, yielding 44 sessions. Two MATLAB formats and several motion-energy wrapper layouts motivated the two loaders.

## 1-b. How are the data split into subjects?

i. The animal string in each registry tuple defines the subject. Unique IDs are sorted, and every retained session receives its index into that list.

ii.
```python
all_animals.append(anm)
subjects_set.add(anm)
subjects = sorted(subjects_set)
subject_idx = np.array([subjects.index(anm) for anm in all_animals])
```

iii. The notes report 14 subjects and use the loading-script animal IDs as authoritative.

## 1-c. How are the data split into sessions?

i. Each registry tuple and corresponding MATLAB file is one session and becomes one element of each top-level session list. Sessions with fewer than two valid trials or fewer than ten retained neurons are skipped.

ii.
```python
for session_idx, (anm, date, probes, data_dir) in enumerate(sessions):
    result = process_session(anm, date, probes, data_dir, ...)
    if result is None:
        continue
    all_neural.append(result['neural'])
```

iii. This follows the authors' per-session files; the ten-unit cutoff was taken from the paper. In practice all 44 registered sessions were retained.

## 1-d. How are the data split into trials?

i. `bp.Ntrials` defines trial count. Per-trial behavior arrays are indexed by zero-based trial index; spike records use one-based `clu.trial`. Each retained trial becomes one neural, input, and output array.

ii.
```python
ntrials_total = int(bp['Ntrials'])
trial_num = j + 1
spk_mask = trial_arr == trial_num
for t_idx in range(n_valid):
    neural_trials.append(trialdat_valid[:, :, t_idx].T)
```

iii. The agent treated Bpod trial rows and spike trial labels as explicit boundaries, avoiding inferred segmentation.

## 1-e. How are trials filtered based on quality controls?

i. Early-lick, photostimulation, and no-response/ignore trials are removed; a redundant `(hit | miss)` condition also enforces a response. It does not drop trials beyond the electrophysiology recording and explicitly retains known all-zero neural trials.

ii.
```python
valid_mask = ~early & ~stim_enable & ~no
valid_mask = valid_mask & (hit | miss)
valid_trials = np.where(valid_mask)[0]
```

iii. The notes claim early, stimulation, and ignore exclusion follows the paper. They acknowledge 30 end-of-recording all-zero trials but retain them “for completeness.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from selected `obj.clu` probes: cluster `quality`, spike `trial`, and spike `trialtm`, together with per-trial `bp.ev.goCue`.

ii.
```python
trial_arr = np.array(clu_probe['trial'][clu_idx]).flatten()
trialtm_arr = np.array(clu_probe['trialtm'][clu_idx]).flatten()
aligned = trialtm_arr[spk_mask] - align_times_all[j]
```

iii. The notes identify these as the fields used by the authors' `findClusters`, `alignSpikes`, and `getSeq` pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned, histogrammed into 10 ms bins, converted to Hz, and passed through a custom causal 15-bin Gaussian smoother with reflected prefix padding. Selected probes are concatenated.

ii.
```python
counts, _ = np.histogram(aligned, bins=edges)
fr = counts.astype(np.float32) / DT
trialdat[:, neuron_offset + i, j] = causal_gaussian_smooth(
    fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. The agent chose parameters from `WorkingWithDataObjs.m`, describing 10 ms, width 15, and reflect as a valid reference configuration. Its custom causal interpretation was intended to match `mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters labelled garbage, gabrga, noisy, or real? (case-insensitive) are excluded. Units must have mean rate strictly above 1 Hz across the full window and all raw trials; sessions then need at least ten units.

ii.
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
if q_stripped.lower() not in {e.lower() for e in excluded_qualities}:
    valid.append(i)
mean_fr = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
keep = mean_fr > low_fr
```

iii. The exclusions follow `findClusters.m`; >1 Hz and the ten-unit session minimum are justified from the paper. Unlike the human solution it does not exclude `poor`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's within-trial time has that trial's go-cue time subtracted before binning over −2.5 to +2.5 seconds.

ii.
```python
align_times_all = goCue
aligned = trialtm_arr[spk_mask] - align_times_all[j]
edges = np.arange(TMIN, TMAX + DT, DT)
```

iii. This is explicitly described as matching the authors' `alignSpikes.m` with `alignEvent='goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses 500 non-overlapping 10 ms bins over five seconds. Raw spikes are binned directly; video streams are linearly interpolated onto the same bin-center grid.

ii.
```python
DT = 1.0 / 100
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
```

iii. The notes recognized both 5 and 10 ms reference usages and selected 10 ms from `WorkingWithDataObjs.m`. The human conversion instead selected the paper pipeline's 5 ms grid.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is a constructed grid of bin centers relative to `bp.ev.goCue`, not a separately sampled raw variable.

ii.
```python
time_axis = edges[:-1] + DT / 2
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. The agent regarded the aligned window itself as the required continuous decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Bin edges from −2.5 to +2.5 seconds at 10 ms spacing are converted to centers by adding 5 ms; the identical row is copied to every trial.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
```

iii. The notes say this matches the time-axis convention in `getSeq.m`.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the centers of exactly the edges used to histogram go-cue-relative spikes, so input column k labels neural bin k.

ii.
```python
counts, _ = np.histogram(aligned, bins=edges)
time_axis = edges[:-1] + DT / 2
```

iii. No additional alignment was considered necessary because both are generated from one common grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived from Bpod `R`, `L`, `hit`, and `miss` flags after ignore trials have been removed.

ii.
```python
R = np.array(bp['R']).flatten().astype(bool)
L = np.array(bp['L']).flatten().astype(bool)
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
```

iii. The notes correctly reason that hit means the instructed direction and miss means the opposite direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Right is encoded 1 for right-hit or left-miss; all retained alternatives are left, encoded 0. No “none” class is produced because ignores are filtered.

ii.
```python
lick_right = (R & hit) | (L & miss)
lick_direction = lick_right[valid_trials].astype(np.int32)
```

iii. The binary direction formula is justified correctly, but the decision to filter ignores contradicts the requested left/right/none output.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context comes directly from `bp.autowater`.

ii.
```python
autowater = np.array(bp['autowater']).flatten().astype(bool)
```

iii. The notes identify automatic uncued water as the WC context marker.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater trials map to WC=0 and all others to DR=1, then the scalar class is broadcast across time.

ii.
```python
context = (~autowater[valid_trials]).astype(np.int32)
out[1, :] = context[t_idx]
```

iii. This direct relabelling follows the prompt and reference mapping.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The code reads `bp.hit`, `bp.miss`, and `bp.no`, though after filtering only hit/miss remain and the stored value is derived solely from `hit`.

ii.
```python
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
no = np.array(bp['no']).flatten().astype(bool)
outcome = hit[valid_trials].astype(np.int32)
```

iii. The notes planned and implemented only correct/incorrect after electing to discard no-response trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hits map to correct=1 and misses to incorrect=0; no ignore=2 category exists because ignores are removed. Values are broadcast across each trial's time bins.

ii.
```python
outcome = hit[valid_trials].astype(np.int32)
out[2, :] = outcome[t_idx]
```

iii. The binary mapping is sound for responded trials, but omitting the explicitly requested ignore class is unjustified relative to the decoder specification.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses only the bottom-camera `obj.traj[1]` DLC `top_tongue` (or a fallback feature containing “tongue”), its x/y coordinates and frame times, plus bitcode timing and go cues.

ii.
```python
traj_bottom = obj['traj'][1]
if name == 'top_tongue': tongue_idx = i
x = ts[:, 0, tongue_idx].copy()
y = ts[:, 1, tongue_idx].copy()
```

iii. The notes call this the tip-of-tongue bottom-camera signal. The human solution combines side and bottom views to improve visibility.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code gradients raw x/y samples at an assumed 400 Hz, takes Euclidean speed, marks originally invalid coordinates NaN, linearly interpolates speed to the neural grid, and finally changes all NaNs to zero. It does not likelihood-filter, smooth coordinates, segment visible runs, normalize views, or combine cameras.

ii.
```python
valid = ~np.isnan(x) & ~np.isnan(y)
vx_all = np.gradient(x) * VIDEO_FR
vy_all = np.gradient(y) * VIDEO_FR
speed = np.sqrt(vx_all**2 + vy_all**2)
speed[~valid] = np.nan
tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)
```

iii. The notes emphasize not nearest-filling tongue coordinates, citing the methods, but then equate invisibility with zero movement; this loses the missingness required by the prompt.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A single median over every retained trial and time bin in the session is used. Below is 0 and at/above is 1. If the median is zero it is replaced by epsilon. There is no class 2 for not visible.

ii.
```python
threshold = np.percentile(all_vals, percentile)
if threshold == 0:
    threshold = np.finfo(np.float32).eps
result = (data_2d >= threshold).astype(np.int32)
```

iii. The median is from the prompt. The epsilon special case was meant to keep exact zero low, but because missing tongue samples were zeroed it cannot implement “2: not visible.”

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Bottom-camera frame time is corrected by a session bitcode offset and the trial go cue, then linearly interpolated at neural bin centers.

ii.
```python
old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, speed, kind='linear',
                    bounds_error=False, fill_value=np.nan)
tongue_vel[:, trix] = f_interp(time_axis)
```

iii. The offset formula is documented as matching `findVideoOffset.m`; shared target times ensure nominal neural/video alignment.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses all bottom-camera DLC features whose names contain “paw,” rather than only the reliably tracked `top_paw`, along with frame timing, bitcode, and go cues.

ii.
```python
traj_bottom = obj['traj'][1]
if 'paw' in name.lower():
    paw_indices.append(i)
```

iii. The notes planned “paw position from bottom cam” and chose to average paw features; they did not document the reliability distinction identified by the human solution.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. NaN coordinates are nearest-filled, raw coordinate gradients are multiplied by 400 Hz, speeds are averaged across both paw features, interpolated onto the neural grid, and remaining missing time bins are nearest-filled.

ii.
```python
arr[nans] = arr[valid[nearest]]
vx = np.gradient(x) * VIDEO_FR
vy = np.gradient(y) * VIDEO_FR
speeds.append(np.sqrt(vx**2 + vy**2))
avg_speed = np.mean(speeds, axis=0)
```

iii. The notes cite nearest filling as the intended treatment for paw data, but do not justify averaging an unreliable second paw or omitting smoothing/likelihood gating.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The session-wide median over retained bins defines 0 below and 1 at/above, with zero median replaced by epsilon. Missing observations are filled before discretization, so class 2 is never represented.

ii.
```python
paw_vel_disc = discretize_per_session(paw_vel_valid)
result = (data_2d >= threshold).astype(np.int32)
```

iii. The 50th percentile follows the prompt, but the output vocabulary is incorrectly binary despite the specified “2: not visible.”

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times are offset-corrected, made relative to the trial go cue, and paw speed is linearly interpolated onto the neural time centers.

ii.
```python
old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, avg_speed, kind='linear',
                    bounds_error=False, fill_value=np.nan)
paw_vel[:, trix] = f_interp(time_axis)
```

iii. This uses the same session offset and common grid as the other video streams.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It loads the standalone `motionEnergy_<animal>_<date>.mat` `me.data` trace per trial, supporting direct, struct, and nested-struct layouts. Side-camera frame times provide timing.

ii.
```python
me_file = scipy.io.loadmat(me_path)
me_var = me_file['me']
me_trial = np.array(me_raw[trix, 0]).flatten()
traj_view0 = obj['traj'][0]
```

iii. The notes identified the standalone motion-energy files and their heterogeneous wrappers from the authors' loader.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The already reduced per-frame signal is linearly interpolated to neural times, out-of-range NaNs are nearest-filled, and the result is discretized. The file's `moveThresh` is loaded but unused.

ii.
```python
f_interp = interp1d(old_time, me_trial, kind='linear',
                    bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_interp(time_axis)
col[nans] = col[valid_idx[nearest]]
```

iii. The notes say spatial motion-energy processing was already done and cite the reference's interpolation/fillmissing behavior.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. It uses the 50th percentile pooled over all retained bins in the session; below is 0 and at/above is 1. Missing files yield all zeros, not class 2.

ii.
```python
me_disc = discretize_per_session(me_valid)
else:
    me_disc = np.zeros((n_time, n_valid), dtype=np.int32)
```

iii. The agent correctly follows the requested median rather than `moveThresh`, but does not implement the requested “2: no video” category.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are corrected by the bitcode-derived session offset and trial go cue, then the signal is interpolated at neural bin centers. A synthetic 400 Hz fallback is used on exceptions.

ii.
```python
old_time = ft - vidshift - align_times[trix]
# fallback
ft = np.arange(1, nframes + 1) / VIDEO_FR
old_time = ft - 0.5 - align_times[trix]
```

iii. The normal path follows `loadMotionEnergy.m`; the fallback was intended to tolerate malformed frame-time data but invents timing rather than marking it unavailable.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The custom v5 loader tolerates odd fields and absent stim data; motion-energy layouts are unwrapped conditionally. Missing tongue becomes zero, paw and motion-energy gaps are nearest-filled, missing motion-energy files become zeros, and timing exceptions use synthetic frames. Known all-zero neural end trials remain. Warnings are broadly suppressed only for `FutureWarning`.

ii.
```python
bp['stim'] = {'enable': np.zeros(int(bp_raw['Ntrials'].flat[0]))}
tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)
col[nans] = col[valid_idx[nearest]]
me_disc = np.zeros((n_time, n_valid), dtype=np.int32)
```

iii. The notes frame these as robust format handling and paper-matching fill policies. However, they explicitly acknowledge retaining recording-artifact trials and do not preserve missing-video categories.

## 11-a. What are the most time-consuming steps of the code?

i. Loading large MATLAB files and, especially, nested neuron-by-trial spike binning/smoothing dominate. The agent times file loading, spike binning, behavior processing, and total session runtime; full conversion took about 300 seconds.

ii.
```python
for probe_idx, valid_clu in all_cluster_indices:
    for i, clu_idx in enumerate(valid_clu):
        for j in range(ntrials_total):
            ...
            causal_gaussian_smooth(...)
```

iii. The notes report the full runtime but do not provide an aggregate profile; the explicit timers and nested loops support this assessment.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The cluster-by-trial spike loop could use a 2-D histogram over trial IDs and aligned times, as in the human solution. Smoothing loops over neuron columns and video loops over trials/features also offer partial vectorization, while ragged frames constrain complete vectorization.

ii.
```python
for i, clu_idx in enumerate(valid_clu):
    for j in range(ntrials_total):
        spk_mask = trial_arr == j + 1
        counts, _ = np.histogram(aligned, bins=edges)
```

iii. The agent claimed “process spikes for all neurons at once,” but the implementation still uses nested Python loops. Its notes do not critically identify this inefficiency.

## 11-c. What processing does the code repeat multiple times?

i. It defines two largely duplicate spike-processing paths (`align_and_bin_spikes` and inline `process_session`) and two low-rate filters, though only the inline versions are used. Per trial it repeatedly parses feature names/arrays and constructs identical input/output shapes; nearest-fill logic is duplicated for paw and motion energy.

ii.
```python
def align_and_bin_spikes(...): ...
def remove_low_fr_clusters(...): ...
# later, duplicated inline spike loop
def remove_low_fr_neurons(...): ...
```

iii. The notes do not acknowledge these duplicate/dead helper implementations; they arose during script development.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads many unused MATLAB fields, computes and returns `me_thresh` without using it, creates unused `vx`/`vy` tongue arrays, processes all raw trials before selecting valid trials, and imports `glob`/`sys` without use. Plotting is optional and therefore not part of normal conversion.

ii.
```python
me_thresh = float(me_struct['moveThresh'].flat[0])
vx = np.full_like(x, np.nan)
vy = np.full_like(y, np.nan)
trialdat_valid = trialdat[:, :, valid_trials]
```

iii. The agent's documentation does not discuss discarded work. Some all-trial processing simplifies session-level filtering, but video processing of excluded trials and unused variables are avoidable.
