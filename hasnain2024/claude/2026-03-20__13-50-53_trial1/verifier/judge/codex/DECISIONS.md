# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the 44 analyzed sessions into two Python lists, `EPHYS_SESSIONS` and `RANDOMIZED_DELAY_SESSIONS`, then iterates that registry in `main()`. For each session it loads `data_structure_<anm>_<date>.mat` from the declared subdirectory under `/app/data`, tries `mat73.loadmat()` first, falls back to a custom `_load_v5_session()` parser for MATLAB v5 files, and separately loads `motionEnergy_<anm>_<date>.mat` if present.

ii. ```python
EPHYS_SESSIONS = [
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ...
]
RANDOMIZED_DELAY_SESSIONS = [
    ('JEB11', '2022-05-10', [1], 'RandomizedDelay_Ephys_Behavior'),
    ...
]
ALL_SESSIONS = EPHYS_SESSIONS + RANDOMIZED_DELAY_SESSIONS
```

```python
def load_session_data(anm, date, data_dir):
    data_path = os.path.join(DATA_ROOT, data_dir, f'data_structure_{anm}_{date}.mat')
    try:
        obj = mat73.loadmat(data_path)['obj']
    except TypeError:
        obj = _load_v5_session(data_path)

    me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{anm}_{date}.mat')
    ...
    return obj, me_raw, me_thresh, t_load
```

iii. The justification in `CONVERSION_NOTES.md` is that the session list should match the paper's loading scripts rather than globbing raw files, and that both MATLAB v7.3 and v5 readers are needed because both formats occur in the dataset.

## 1-b. How are the data split into subjects (mice)?

i. The AI treats the `anm` field from each hard-coded session tuple as the subject id. It accumulates unique animal ids into a sorted `subjects` list and stores one `subject_idx` per kept session.

ii. ```python
for i, (anm, date, probes, data_dir) in enumerate(sessions):
    ...
    all_animals.append(anm)
    subjects_set.add(anm)
```

```python
subjects = sorted(subjects_set)
subject_idx = np.array([subjects.index(anm) for anm in all_animals], dtype=np.int64)
```

iii. The notes justify this by listing the included mice from the loading scripts and treating session tuples as authoritative session metadata.

## 1-c. How are the data split into sessions?

i. Each tuple `(anm, date, probes, data_dir)` is one session. `main()` loops over `ALL_SESSIONS`, calls `process_session()` once per tuple, and appends each result as one entry in `neural`, `input`, and `output`.

ii. ```python
for i, (anm, date, probes, data_dir) in enumerate(sessions):
    result = process_session(anm, date, probes, data_dir, ...)
    ...
    all_neural.append(result['neural'])
    all_input.append(result['input'])
    all_output.append(result['output'])
```

iii. The AI's notes say it is using the sessions from the paper's per-animal loading scripts, including 25 fixed-delay and 19 randomized-delay sessions.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the per-trial behavioral arrays in `obj['bp']`, with `ntrials_total = int(bp['Ntrials'])`. The AI creates a boolean `valid_mask` over trial indices, converts it to `valid_trials = np.where(valid_mask)[0]`, and then slices neural and output arrays by those trial indices. Within neural data, spike events are assigned to trials using each cluster's `trial` field.

ii. ```python
bp = obj['bp']
ntrials_total = int(bp['Ntrials'])
...
valid_trials = np.where(valid_mask)[0]
```

```python
trial_arr = np.array(clu_probe['trial'][clu_idx]).flatten()
...
for j in range(ntrials_total):
    trial_num = j + 1
    spk_mask = trial_arr == trial_num
```

iii. The notes describe the raw data as one session object with one entry per trial across behavior, video, and motion-energy arrays; there is no separate justification beyond using the dataset's own trial indexing.

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes early-lick trials, photostimulation trials, and no-response/ignore trials. It also requires `hit | miss`, so only response trials are kept. It does not drop post-recording tail trials with no spikes; the notes instead mention retaining some all-zero end-of-session trials "for completeness."

ii. ```python
no = np.array(bp['no']).flatten().astype(bool)
autowater = np.array(bp['autowater']).flatten().astype(bool)
early = np.array(bp['early']).flatten().astype(bool)
...
valid_mask = ~early & ~stim_enable & ~no
valid_mask = valid_mask & (hit | miss)
valid_trials = np.where(valid_mask)[0]
```

iii. `CONVERSION_NOTES.md` explicitly says: "Trial filtering: exclude early, stim.enable, and no-response trials. Keep hit and miss." The notes also acknowledge sessions with all-zero neural trials at the end and say those trials were retained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `obj['clu']` for the selected probes, using each cluster's `trial`, `trialtm`, and `quality` fields. Alignment uses `obj['bp']['ev']['goCue']`.

ii. ```python
trial_arr = np.array(clu_probe['trial'][clu_idx]).flatten()
trialtm_arr = np.array(clu_probe['trialtm'][clu_idx]).flatten()
...
goCue = np.array(ev['goCue']).flatten()
```

```python
qualities = clu_probe['quality']
...
if q_stripped.lower() not in {e.lower() for e in excluded_qualities}:
    valid.append(i)
```

iii. The notes identify `alignSpikes`, `getSeq`, and `findClusters` from the MATLAB code as the main reference for using spike times, alignment times, and cluster quality labels.

## 2-b. How is the `neural` data processed?

i. For each kept cluster and each trial, the AI subtracts the go cue from `trialtm`, bins spikes from `-2.5` to `2.5` s in `DT = 1/100` second bins, converts counts to firing rate by dividing by `DT`, and applies a custom causal Gaussian smoothing kernel of width 15 bins with reflect padding. The final per-trial arrays are stored as `(n_neurons, n_time)` `float32` matrices.

ii. ```python
DT = 1.0 / 100  # 10 ms bins
SMOOTH_WINDOW = 15  # bins, causal Gaussian
```

```python
aligned = trialtm_arr[spk_mask] - align_times_all[j]
counts, _ = np.histogram(aligned, bins=edges)
fr = counts.astype(np.float32) / DT
trialdat[:, neuron_offset + i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. The notes justify this as matching `alignSpikes`, `getSeq`, and `mySmooth.m`, and explicitly say the AI resolved the dt discrepancy by choosing 10 ms bins from `WorkingWithDataObjs.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first excludes clusters whose lower-cased quality label is in `{'garbage', 'gabrga', 'noisy', 'real?'}`. After spike binning and smoothing, it removes neurons whose mean firing rate across time and trials is `<= 1 Hz`. It also skips any session that ends up with fewer than 10 neurons after this filter.

ii. ```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0
```

```python
if q_stripped.lower() not in {e.lower() for e in excluded_qualities}:
    valid.append(i)
...
mean_fr = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
keep = mean_fr > low_fr
...
if n_neurons_final < 10:
    return None
```

iii. `CONVERSION_NOTES.md` says the AI adopted the paper's 1 Hz threshold and the cluster labels excluded by `findClusters.m`; it additionally cites the paper's "at least 10 units" statement when describing session-quality expectations.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned by subtracting each trial's go-cue time from the spike times assigned to that trial.

ii. ```python
goCue = np.array(ev['goCue']).flatten()
...
aligned = trialtm_arr[spk_mask] - align_times_all[j]
```

iii. The notes explicitly connect this to the reference MATLAB statement `trialtm_aligned = trialtm - goCue_time`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins (`DT = 1/100`) over `[-2.5, 2.5]` s, producing 500 bins per trial. There is no later rebinning; this bin grid is used directly for neural, input, and interpolated behavioral streams.

ii. ```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 100  # 10 ms bins
...
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
```

iii. The notes state that the AI chose 10 ms because it considered `WorkingWithDataObjs.m` more authoritative than scripts using 5 ms bins.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The AI derives this input from the global aligned time grid defined around the go cue, rather than from a separate raw data variable. The only raw variable involved is the alignment event `bp.ev.goCue`, which motivates the time origin.

ii. ```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
...
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. The notes say this variable is simply "time from goCue (s)" and is included as the decoder input over the same aligned analysis window.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes bin centers from the fixed `[-2.5, 2.5]` s window and the 10 ms bin size, then reuses that same `time_axis` for every trial in every session.

ii. ```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
...
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. There is no further justification in the trajectory beyond using the common decoder time axis.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is exactly the same `time_axis` used for spike binning, so each input sample corresponds to the center of the neural time bin at that column.

ii. ```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
...
counts, _ = np.histogram(aligned, bins=edges)
...
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. The notes justify this by treating time from go cue as the common alignment coordinate for the entire converted dataset.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI uses the per-trial behavioral flags `bp.R`, `bp.L`, `bp.hit`, and `bp.miss`. It also reads `bp.no` because no-response trials are removed before lick direction is assigned.

ii. ```python
R = np.array(bp['R']).flatten().astype(bool)
L = np.array(bp['L']).flatten().astype(bool)
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
no = np.array(bp['no']).flatten().astype(bool)
```

iii. The notes map lick direction to `obj.bp.R/L + hit/miss` and explain that no-response trials are excluded rather than represented as a separate class.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. After filtering to hit/miss trials, the AI assigns rightward licks when `(R & hit) | (L & miss)` is true and leftward licks otherwise. It encodes only two classes, `left=0` and `right=1`, and broadcasts the per-trial label across all time bins.

ii. ```python
lick_right = (R & hit) | (L & miss)
lick_direction = lick_right[valid_trials].astype(np.int32)
...
out[0, :] = lick_direction[t_idx]
```

iii. `CONVERSION_NOTES.md` states the same mapping and explicitly says no-response trials are excluded.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The AI derives context directly from `bp.autowater`.

ii. ```python
autowater = np.array(bp['autowater']).flatten().astype(bool)
```

iii. The notes describe the mapping as `autowater=1 -> WC`, otherwise `DR`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI encodes `WC` as `0` and `DR` as `1` by taking the logical negation of `autowater`, then broadcasts that per-trial label across time bins.

ii. ```python
context = (~autowater[valid_trials]).astype(np.int32)
...
out[1, :] = context[t_idx]
```

iii. The notes justify the coding as matching the instruction's WC=0, DR=1 convention.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI derives outcome from `bp.hit` and `bp.miss`; `bp.no` is only used upstream to discard ignore trials.

ii. ```python
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
no = np.array(bp['no']).flatten().astype(bool)
```

iii. The notes map outcome to `hit -> correct`, `miss -> incorrect`, with ignore/no-response trials excluded.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI uses a two-class outcome label after trial filtering: `correct=1` for hit trials and `incorrect=0` for miss trials. It does not include an ignore class because those trials were removed.

ii. ```python
outcome = hit[valid_trials].astype(np.int32)
...
out[2, :] = outcome[t_idx]
```

iii. `CONVERSION_NOTES.md` explicitly says the conversion keeps hit and miss only and excludes no-response trials.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI derives tongue velocity from the bottom camera (`obj['traj'][1]`) DeepLabCut traces, primarily the feature named `top_tongue`, with a fallback to the first feature containing `"tongue"`. It also uses bottom-camera `frameTimes`, the session-wide video offset, and the trial go-cue times for alignment.

ii. ```python
traj_bottom = obj['traj'][1]  # bottom cam
feat_names_raw = traj_bottom['featNames'][0]
...
if name == 'top_tongue':
    tongue_idx = i
```

```python
ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
```

iii. The notes justify this as using "tip-of-tongue displacement from bottom cam view" for tongue velocity.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the AI takes the bottom-camera tongue x/y coordinates, computes framewise gradients at 400 Hz without smoothing, converts them to speed magnitude, masks out frames where x or y are NaN, linearly interpolates the speed trace onto the neural time axis, and finally replaces NaNs with zeros. It then discretizes the aligned session-wide values at the 50th percentile.

ii. ```python
x = ts[:, 0, tongue_idx].copy()
y = ts[:, 1, tongue_idx].copy()
...
vx_all = np.gradient(x) * VIDEO_FR
vy_all = np.gradient(y) * VIDEO_FR
speed = np.sqrt(vx_all**2 + vy_all**2)
speed[~valid] = np.nan
...
f_interp = interp1d(old_time, speed, kind='linear',
                   bounds_error=False, fill_value=np.nan)
tongue_vel[:, trix] = f_interp(time_axis)
...
tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)
```

iii. The notes say the AI intended to avoid nearest-filling tongue NaNs "per paper methods," but the final code still turns missing aligned bins into 0 before thresholding.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI uses `discretize_per_session()` with a 50th-percentile threshold over all finite tongue-velocity samples in the session. Values below threshold become `0`, values at or above threshold become `1`. If the threshold is exactly zero, it is replaced with machine epsilon so zeros stay in the low bin; NaN positions are assigned `0`.

ii. ```python
def discretize_per_session(data_2d, percentile=50):
    all_vals = data_2d[~np.isnan(data_2d)]
    threshold = np.percentile(all_vals, percentile)
    if threshold == 0:
        threshold = np.finfo(np.float32).eps
    result = (data_2d >= threshold).astype(np.int32)
    result[np.isnan(data_2d)] = 0
    return result
```

```python
tongue_vel_disc = discretize_per_session(tongue_vel_valid)
```

iii. The notes justify the median split from the prompt and describe missing tongue bins as effectively treated as no movement.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI computes a session-wide video offset from SpikeGLX bitcodes and Bpod bit-start times, subtracts that offset and each trial's go cue from bottom-camera `frameTimes`, then linearly interpolates tongue speed onto the neural `time_axis`.

ii. ```python
def compute_video_offset(obj):
    bitStart = scipy_stats.mode(np.array(obj['bp']['ev']['bitStart']).flatten(), keepdims=False).mode
    bc_bitstart = np.array(obj['sglx']['bitcode']['bitstart']).flatten()
    bc_mode = scipy_stats.mode(bc_bitstart, keepdims=False).mode
    fs = float(obj['sglx']['fs'])
    vidshift = bc_mode / fs - bitStart
    return vidshift
```

```python
old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, speed, kind='linear',
                   bounds_error=False, fill_value=np.nan)
tongue_vel[:, trix] = f_interp(time_axis)
```

iii. The notes cite `findVideoOffset.m` as the reference for the session-wide offset computation.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The AI derives paw velocity from the bottom camera (`obj['traj'][1]`) and uses every bottom-camera feature whose name contains `"paw"`, so it may combine both `top_paw` and `bottom_paw`. It also uses `frameTimes`, the video offset, and go-cue times.

ii. ```python
traj_bottom = obj['traj'][1]  # bottom cam
...
paw_indices = []
for i, name in enumerate(feat_names):
    if 'paw' in name.lower():
        paw_indices.append(i)
```

iii. The notes describe this more generically as using paw position from the bottom camera to compute velocity.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature on each trial, the AI nearest-fills NaNs in x and y, computes framewise gradients at 400 Hz, converts them to speed magnitude, averages speeds across all detected paw features, interpolates the result onto the neural time axis, and nearest-fills any remaining NaNs in the aligned trace. It then median-splits the session.

ii. ```python
for arr in [x, y]:
    nans = np.isnan(arr)
    if nans.any() and not nans.all():
        valid = np.where(~nans)[0]
        nan_pos = np.where(nans)[0]
        nearest = np.searchsorted(valid, nan_pos).clip(0, len(valid)-1)
        arr[nans] = arr[valid[nearest]]

vx = np.gradient(x) * VIDEO_FR
vy = np.gradient(y) * VIDEO_FR
speeds.append(np.sqrt(vx**2 + vy**2))
...
avg_speed = np.mean(speeds, axis=0)
```

```python
f_interp = interp1d(old_time, avg_speed, kind='linear',
                   bounds_error=False, fill_value=np.nan)
...
col[nans] = col[valid_idx[nearest]]
```

iii. `CONVERSION_NOTES.md` explicitly says the AI decided to fill paw NaNs with nearest neighbors before velocity computation.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is discretized with the same session-level 50th-percentile split used for tongue velocity: below threshold `0`, at/above threshold `1`, with NaNs mapped to `0`.

ii. ```python
paw_vel_valid = paw_vel[:, valid_trials]
paw_vel_disc = discretize_per_session(paw_vel_valid)
```

iii. The notes justify this as following the prompt's requested per-session median split.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity is aligned by subtracting the video offset and the trial go cue from bottom-camera `frameTimes`, then linearly interpolating the resulting speed trace onto the neural time axis.

ii. ```python
ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
...
f_interp = interp1d(old_time, avg_speed, kind='linear',
                   bounds_error=False, fill_value=np.nan)
paw_vel[:, trix] = f_interp(time_axis)
```

iii. The AI's justification is implicit: it reused the same bitcode-based video offset and neural time grid as for motion energy and tongue velocity.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from the separate `motionEnergy_<anm>_<date>.mat` file. The AI unwraps several file layouts to obtain `me_raw`, which it treats as one motion-energy trace per trial, and aligns it using side-camera `frameTimes` from `obj['traj'][0]`.

ii. ```python
me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{anm}_{date}.mat')
...
if me_var.dtype.names:
    me_struct = me_var[0, 0]
    me_data = me_struct['data']
    if me_data.dtype.names and 'data' in me_data.dtype.names:
        inner = me_data[0, 0]
        me_raw = inner['data']
```

```python
traj_view0 = obj['traj'][0]  # side cam
me_trial = np.array(me_raw[trix, 0]).flatten()
```

iii. The notes justify this by citing the reference motion-energy loader and noting that multiple `.mat` layouts occur across sessions.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI linearly interpolates each per-frame motion-energy trace onto the neural time axis after video-offset correction, then nearest-fills NaNs within each aligned trial and discretizes the values at the per-session 50th percentile.

ii. ```python
old_time = ft - vidshift - align_times[trix]
...
f_interp = interp1d(old_time, me_trial, kind='linear',
                   bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_interp(time_axis)
```

```python
for trix in range(ntrials):
    col = me_aligned[:, trix]
    nans = np.isnan(col)
    if nans.any() and not nans.all():
        ...
        col[nans] = col[valid_idx[nearest]]
```

iii. The notes say this was intended to match `loadMotionEnergy.m`, including handling nested `me.data.data`, although the final implementation uses interpolation and nearest-fill rather than the reference solution's frame-to-bin averaging.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy uses the same `discretize_per_session()` function as tongue and paw velocity: a session-wise median split into 0/1, with NaNs forced to 0.

ii. ```python
me_valid = me_aligned[:, valid_trials]
me_disc = discretize_per_session(me_valid)
```

iii. The notes justify the threshold choice as following the prompt's 50th-percentile discretization instruction.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI uses side-camera `frameTimes`, subtracts the session-wide video offset and the trial go cue, then linearly interpolates the motion-energy trace onto the neural `time_axis`.

ii. ```python
traj_view0 = obj['traj'][0]  # side cam
...
ft = np.array(traj_view0['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, me_trial, kind='linear',
                   bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_interp(time_axis)
```

iii. The justification given in the notes is that motion energy should be aligned through the same video-offset correction used for camera-derived features.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses several fallback and imputation rules. For MATLAB v5 files it uses a custom parser and try/except conversions for problematic fields. If motion-energy or video frame-times are missing or unreadable, it fabricates a 400 Hz frame-time axis. Tongue NaNs survive gradient computation but are later turned into 0. Paw NaNs are nearest-filled both before and after interpolation. Motion-energy NaNs are nearest-filled after interpolation. It also retains some end-of-session trials with all-zero neural activity instead of dropping them.

ii. ```python
except TypeError:
    obj = _load_v5_session(data_path)
...
except:
    nframes = len(me_trial)
    ft = np.arange(1, nframes + 1) / VIDEO_FR
    old_time = ft - 0.5 - align_times[trix]
```

```python
tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)
...
arr[nans] = arr[valid[nearest]]
...
col[nans] = col[valid_idx[nearest]]
```

iii. The notes justify these as practical handling for mixed file layouts and missing tracking values; they explicitly say tongue NaNs were intentionally not nearest-filled, but the code still maps aligned missing tongue bins to zero.

## 11-a. What are the most time-consuming steps of the code?

i. The AI instrumented each session with load time, spike-binning time, and behavioral-processing time. In the full-run log, session loading is usually the dominant cost, spike binning is second, and behavioral processing is much smaller.

ii. ```python
t0 = time.time()
...
t_load = time.time() - t0
...
t_bin = time.time() - t_bin_start
...
t_behav = time.time() - t_behav_start
print(f"  Loaded in {t_load:.1f}s")
print(f"  Spike binning: {t_bin:.1f}s for {total_neurons} neurons")
print(f"  Behavioral processing: {t_behav:.1f}s")
```

iii. There is no separate prose justification in the notes beyond the timing printouts; the observed logs show loading dominates most sessions.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The main avoidable hot loops are the nested `for cluster -> for trial` spike-binning loops in `process_session()`, plus the per-trial interpolation loops for tongue velocity, paw velocity, and motion energy. There is also an unused helper, `align_and_bin_spikes()`, that contains a second copy of the same cluster/trial loop structure.

ii. ```python
for probe_idx, valid_clu in all_cluster_indices:
    ...
    for i, clu_idx in enumerate(valid_clu):
        ...
        for j in range(ntrials_total):
            trial_num = j + 1
            spk_mask = trial_arr == trial_num
            ...
            counts, _ = np.histogram(aligned, bins=edges)
```

```python
for trix in range(ntrials):
    ...
    f_interp = interp1d(old_time, ..., kind='linear',
                       bounds_error=False, fill_value=np.nan)
```

iii. The AI did not provide an explicit justification for leaving these loops unvectorized; the code comments instead emphasize correctness and timing visibility.

## 11-c. What processing does the code repeat multiple times?

i. The AI repeats the same spike-binning logic in two places: once in the unused `align_and_bin_spikes()` helper and again inline in `process_session()`. It also separately parses bottom-camera `featNames` in both `compute_tongue_velocity()` and `compute_paw_velocity()`, rebuilds the time axis inside each session, and reruns similar nearest-fill logic across motion-energy and paw traces.

ii. ```python
def align_and_bin_spikes(...):
    ...
    for i, clu_idx in enumerate(cluster_indices):
        ...
        for j in range(ntrials):
            ...
```

```python
for probe_idx, valid_clu in all_cluster_indices:
    ...
    for i, clu_idx in enumerate(valid_clu):
        ...
        for j in range(ntrials_total):
            ...
```

iii. No explicit justification was found in the notes or trajectory for this duplication.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes continuous tongue, paw, and motion-energy traces only to discard them after median-thresholding into binary outputs. It loads `me_thresh` from the motion-energy files but never uses it. It also contains plotting code and an unused spike-binning helper that are not part of the saved dataset. More generally, it materializes many raw fields in `_load_v5_session()` that are not used downstream.

ii. ```python
obj, me_raw, me_thresh, t_load = load_session_data(anm, date, data_dir)
...
tongue_vel = compute_tongue_velocity(...)
paw_vel = compute_paw_velocity(...)
...
me_disc = discretize_per_session(me_valid)
```

```python
def plot_processing(...):
    ...

def align_and_bin_spikes(...):
    ...
```

iii. The notes justify plotting as a validation aid for `--show-processing`, but there is no explicit justification for the unused `me_thresh`, the duplicate spike-binning helper, or keeping discarded continuous traces only transiently.
