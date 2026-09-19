# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-coded the session registry as two Python lists, `EPHYS_SESSIONS` and `RANDOMIZED_DELAY_SESSIONS`, then iterated through `ALL_SESSIONS`. For each session it loaded `data_structure_<anm>_<date>.mat` from the session's declared folder with `mat73`, falling back to a custom MATLAB-v5 parser, and loaded `motionEnergy_<anm>_<date>.mat` separately with `scipy.io.loadmat`.

ii. 
```python
EPHYS_SESSIONS = [
    ('EKH1', '2021-08-07', [2], 'Ephys_Behavior'),
    ...
]
RANDOMIZED_DELAY_SESSIONS = [
    ('JEB11', '2022-05-10', [1], 'RandomizedDelay_Ephys_Behavior'),
    ...
]
ALL_SESSIONS = EPHYS_SESSIONS + RANDOMIZED_DELAY_SESSIONS

def load_session_data(anm, date, data_dir):
    data_path = os.path.join(DATA_ROOT, data_dir, f'data_structure_{anm}_{date}.mat')
    try:
        obj = mat73.loadmat(data_path)['obj']
    except TypeError:
        obj = _load_v5_session(data_path)
    ...
    me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{anm}_{date}.mat')
```

iii. In `CONVERSION_NOTES.md`, the AI says the session list came from the paper's loading scripts and that excluded sessions such as `JEB23_2023-10-20`, `JEB24_2023-10-03`, and `JEB24_2023-10-04` were omitted because they were not in those scripts.

## 1-b. How are the data split into subjects?

i. The subject is the `anm` field from each hard-coded session tuple. During assembly the AI collected the unique animal IDs into a sorted `subjects` list and built `subject_idx` from the per-session animal names.

ii. 
```python
for i, (anm, date, probes, data_dir) in enumerate(sessions):
    ...
    all_animals.append(anm)
    subjects_set.add(anm)

subjects = sorted(subjects_set)
subject_idx = np.array([subjects.index(anm) for anm in all_animals], dtype=np.int64)
```

iii. The notes justify this by listing 44 sessions across 14 mice and treating the filename/session metadata as the reliable source of animal identity.

## 1-c. How are the data split into sessions?

i. One hard-coded tuple in `ALL_SESSIONS` is one session. Each session is processed independently by `process_session`, and each successful return becomes one entry in `neural`, `input`, and `output`.

ii. 
```python
for i, (anm, date, probes, data_dir) in enumerate(sessions):
    result = process_session(anm, date, probes, data_dir, ...)
    if result is None:
        continue
    all_neural.append(result['neural'])
    all_input.append(result['input'])
    all_output.append(result['output'])
```

iii. The notes explicitly state that the AI intended to include all 44 electrophysiology sessions from the loading scripts and to skip behavior-only folders.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the trial-wise Bpod arrays in `obj['bp']`. The AI uses `bp['Ntrials']` as the session trial count, flattens per-trial behavioral arrays, computes a boolean `valid_mask` over those trial indices, and then slices neural and behavioral arrays by the resulting `valid_trials` indices.

ii. 
```python
bp = obj['bp']
ntrials_total = int(bp['Ntrials'])
R = np.array(bp['R']).flatten().astype(bool)
hit = np.array(bp['hit']).flatten().astype(bool)
...
valid_trials = np.where(valid_mask)[0]
...
trialdat_valid = trialdat[:, :, valid_trials]
```

iii. The notes frame the raw data as one entry per trial in `obj.bp` and use those arrays as the canonical trial structure. There is no explicit justification for not truncating overlong fields to `Ntrials`.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops early-lick trials, photostimulation trials, and no-response/ignore trials, and then keeps only trials with `hit` or `miss`. It does not drop trials that occur after the recording stops; its notes acknowledge all-zero end-of-session neural trials and say they were retained. It also skips any session with fewer than 2 valid trials or fewer than 10 neurons after firing-rate filtering.

ii. 
```python
valid_mask = ~early & ~stim_enable & ~no
valid_mask = valid_mask & (hit | miss)
valid_trials = np.where(valid_mask)[0]

if n_valid < 2:
    return None
...
if n_neurons_final < 10:
    return None
```

iii. `CONVERSION_NOTES.md` says the AI excluded `early`, `stim.enable`, and `no-response` trials because the decoder should use trials with lick responses, and it kept some all-zero neural trials "for completeness."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from the spike-sorted cluster structure in `obj['clu']`, specifically each cluster's `trial`, `trialtm`, and `quality` fields, together with the per-trial go-cue times in `obj['bp']['ev']['goCue']`.

ii. 
```python
trial_arr = np.array(clu_probe['trial'][clu_idx]).flatten()
trialtm_arr = np.array(clu_probe['trialtm'][clu_idx]).flatten()
...
goCue = np.array(ev['goCue']).flatten()
...
aligned = trialtm_arr[spk_mask] - align_times_all[j]
```

iii. The notes describe the reference neural pipeline as quality filtering, alignment to `goCue`, spike binning, smoothing, and low-firing-rate removal.

## 2-b. How is the `neural` data processed?

i. The AI aligns spikes to go cue, bins them from `-2.5` to `2.5` s in 10 ms bins, converts counts to firing rates by dividing by `DT`, and applies a causal Gaussian smoothing kernel of width 15 bins with reflect padding. No additional normalization or baseline subtraction is applied.

ii. 
```python
DT = 1.0 / 100  # 10 ms bins
SMOOTH_WINDOW = 15  # bins, causal Gaussian

counts, _ = np.histogram(aligned, bins=edges)
fr = counts.astype(np.float32) / DT
trialdat[:, neuron_offset + i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. The notes justify this as matching `WorkingWithDataObjs.m`, which they interpreted as using `dt = 1/100`, `smooth = 15`, and a causal `mySmooth.m` implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps clusters unless their quality label lower-cases to one of `garbage`, `gabrga`, `noisy`, or `real?`, then removes neurons whose mean firing rate is not greater than 1 Hz. It also discards sessions with fewer than 10 neurons after this filter.

ii. 
```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
LOW_FR = 1.0

def get_valid_cluster_indices(clu_probe, excluded_qualities=EXCLUDED_QUALITIES):
    ...
    if q_stripped.lower() not in {e.lower() for e in excluded_qualities}:
        valid.append(i)

mean_fr = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
keep = mean_fr > low_fr
...
if n_neurons_final < 10:
    return None
```

iii. The notes say this choice came from `findClusters.m`, the paper's "firing rates exceeding 1 Hz" rule, and a paper statement that analyzed sessions had at least 10 units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For each trial, the AI subtracts that trial's go-cue time from the cluster's `trialtm` spike times before binning, so all neural data are expressed relative to go-cue onset.

ii. 
```python
align_times_all = goCue
...
aligned = trialtm_arr[spk_mask] - align_times_all[j]
counts, _ = np.histogram(aligned, bins=edges)
```

iii. The notes cite the reference `alignSpikes` logic, `trialtm_aligned = trialtm - goCue_time`, as the intended alignment rule.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins over a `-2.5` to `2.5` s window, yielding 500 bins. No later rebinning is applied.

ii. 
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 100
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
```

iii. The AI justified 10 ms bins in the notes by saying the reference code was inconsistent between `1/100` and `1/200`, and it chose `1/100` from `WorkingWithDataObjs.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a dedicated raw variable. The AI constructs it as the bin-center time axis associated with the go-cue-aligned neural grid.

ii. 
```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
...
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. The notes describe this as "Continuous time axis" ranging from `-2.5` to `2.5` s from go cue.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes a uniformly spaced time vector using the 10 ms neural bin edges and stores that same vector for every trial as a `(1, n_time)` array.

ii. 
```python
time_axis = edges[:-1] + DT / 2
...
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. The notes justify this only as the decoder's required continuous "time from goCue" input.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The AI uses the exact same `time_axis` that it used when binning spikes, so each input timepoint corresponds to the center of the neural bin at that column.

ii. 
```python
time_axis = edges[:-1] + DT / 2
...
counts, _ = np.histogram(aligned, bins=edges)
...
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. The notes treat the time axis as part of the same aligned spike-processing grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj['bp']['R']`, `obj['bp']['L']`, `obj['bp']['hit']`, and `obj['bp']['miss']`, after trial filtering removes ignore trials.

ii. 
```python
R = np.array(bp['R']).flatten().astype(bool)
L = np.array(bp['L']).flatten().astype(bool)
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
...
lick_right = (R & hit) | (L & miss)
```

iii. The notes justify this by mapping instructed side plus hit/miss to actual lick side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI encodes only two lick classes: right if `(R & hit) | (L & miss)`, otherwise left on the remaining valid trials. It removes no-lick/ignore trials upstream instead of giving them a third class.

ii. 
```python
valid_mask = ~early & ~stim_enable & ~no
valid_mask = valid_mask & (hit | miss)
...
lick_right = (R & hit) | (L & miss)
lick_direction = lick_right[valid_trials].astype(np.int32)
...
'output_values': [
    ['left', 'right'],
```

iii. The notes say "exclude early, stim.enable, and no-response trials" and describe lick direction as a left/right per-trial variable.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj['bp']['autowater']`.

ii. 
```python
autowater = np.array(bp['autowater']).flatten().astype(bool)
```

iii. The notes map `autowater=1` to WC and `autowater=0` to DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI directly re-labels `autowater` to context code, using `0` for WC and `1` for DR.

ii. 
```python
context = (~autowater[valid_trials]).astype(np.int32)
...
'output_values': [
    ['left', 'right'],
    ['WC', 'DR'],
```

iii. The notes explicitly state "autowater -> WC(0), ~autowater -> DR(1)."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is effectively derived from `hit`, `miss`, and the `no` flag used during trial filtering. After filtering, the stored outcome becomes a binary correct/incorrect label taken from `hit`.

ii. 
```python
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
no = np.array(bp['no']).flatten().astype(bool)
...
valid_mask = ~early & ~stim_enable & ~no
valid_mask = valid_mask & (hit | miss)
...
outcome = hit[valid_trials].astype(np.int32)
```

iii. The notes justify this by saying no-response trials were excluded and outcome should be hit versus miss on the remaining trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI stores outcome as a binary per-trial label: `1` for `hit` and `0` for `miss`. Ignore trials are not encoded as a third class because they are removed before output construction.

ii. 
```python
outcome = hit[valid_trials].astype(np.int32)
...
'output_values': [
    ['left', 'right'],
    ['WC', 'DR'],
    ['incorrect', 'correct'],
```

iii. The notes describe "hit -> correct(1); miss -> incorrect(0)" and separately state that no-response trials were excluded.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the bottom camera tracking in `obj['traj'][1]`, specifically the `featNames`, `ts`, and `frameTimes` fields. The AI looks for `top_tongue` first, then any feature name containing `tongue`. It also uses `bp.ev.goCue` and the session video offset from `sglx.bitcode.bitstart` and `bp.ev.bitStart`.

ii. 
```python
traj_bottom = obj['traj'][1]
feat_names_raw = traj_bottom['featNames'][0]
...
if name == 'top_tongue':
    tongue_idx = i
...
ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
```

iii. The notes justify this as "Use tip-of-tongue displacement from bottom cam view" and say tongue NaNs should not be nearest-filled.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the AI extracts tongue x/y coordinates from the bottom view, computes framewise gradients at 400 Hz, takes the speed magnitude, interpolates that speed onto the neural time axis, and finally converts NaNs to `0.0`. It does not use an explicit likelihood threshold, run-wise smoothing, or two-camera normalization/averaging.

ii. 
```python
x = ts[:, 0, tongue_idx].copy()
y = ts[:, 1, tongue_idx].copy()
valid = ~np.isnan(x) & ~np.isnan(y)
if np.sum(valid) >= 2:
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

iii. The notes claim this follows the paper's tongue missing-data handling and interpret non-visible tongue frames as "no tongue movement."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI applies a single per-session 50th-percentile threshold to the aligned continuous tongue-speed values and returns only two classes, `0` and `1`. Missing bins become `0` because NaNs are converted to zero before discretization and because `discretize_per_session` maps NaNs to `0`.

ii. 
```python
def discretize_per_session(data_2d, percentile=50):
    all_vals = data_2d[~np.isnan(data_2d)]
    threshold = np.percentile(all_vals, percentile)
    if threshold == 0:
        threshold = np.finfo(np.float32).eps
    result = (data_2d >= threshold).astype(np.int32)
    result[np.isnan(data_2d)] = 0
    return result
...
tongue_vel_disc = discretize_per_session(tongue_vel_valid)
```

iii. The notes justify the 50th-percentile split from the task specification. They do not justify adding a third "not visible" class; instead they explicitly collapse non-visible frames into zero velocity.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI computes a session video offset from `sglx.bitcode.bitstart / fs - mode(bp.ev.bitStart)`, subtracts that and the trial's go cue from bottom-camera `frameTimes`, and linearly interpolates the tongue-speed trace onto the neural `time_axis`.

ii. 
```python
def compute_video_offset(obj):
    bitStart = scipy_stats.mode(np.array(obj['bp']['ev']['bitStart']).flatten(), keepdims=False).mode
    bc_mode = scipy_stats.mode(bc_bitstart, keepdims=False).mode
    fs = float(obj['sglx']['fs'])
    vidshift = bc_mode / fs - bitStart

old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, speed, kind='linear',
                   bounds_error=False, fill_value=np.nan)
tongue_vel[:, trix] = f_interp(time_axis)
```

iii. The notes cite the reference video-offset formula and treat interpolation onto the neural time axis as the alignment step.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom camera in `obj['traj'][1]`. The AI uses every feature whose name contains `paw`, so it can combine both `top_paw` and `bottom_paw`, plus `frameTimes`, go cue, and the session video offset.

ii. 
```python
traj_bottom = obj['traj'][1]
...
for i, name in enumerate(feat_names):
    if 'paw' in name.lower():
        paw_indices.append(i)
```

iii. The notes justify this as using "paw position from bottom cam" and computing speed magnitude from the tracked paw coordinates.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature on each trial, the AI nearest-fills NaNs in x and y, computes framewise gradients at 400 Hz, converts them to speed magnitude, averages speeds across all paw features, then linearly interpolates the result to the neural time axis. Afterwards it nearest-fills NaNs in the aligned trace.

ii. 
```python
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
...
f_interp = interp1d(old_time, avg_speed, kind='linear',
                   bounds_error=False, fill_value=np.nan)
```

iii. The notes explicitly say "Paw NaN values filled with nearest neighbor before velocity computation."

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Like tongue velocity, the AI applies a per-session 50th-percentile threshold and produces only two classes, `0` and `1`. Remaining NaNs are replaced with `0`.

ii. 
```python
paw_vel_valid = paw_vel[:, valid_trials]
paw_vel_disc = discretize_per_session(paw_vel_valid)
...
'output_values': [
    ...
    ['low', 'high'],
```

iii. The notes justify the median split from the decoder specification and do not preserve a distinct not-visible class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The AI aligns paw velocity exactly as it does tongue velocity: bottom-camera frame times are corrected by the session video offset and trial go cue, then interpolated onto the neural `time_axis`.

ii. 
```python
ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
...
f_interp = interp1d(old_time, avg_speed, kind='linear',
                   bounds_error=False, fill_value=np.nan)
paw_vel[:, trix] = f_interp(time_axis)
```

iii. The notes treat this as the standard video-to-neural alignment used across outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<anm>_<date>.mat` file. The AI unwraps either a struct layout or a direct cell array and then aligns each trial's motion-energy trace using side-camera frame times from `obj['traj'][0]['frameTimes']`.

ii. 
```python
me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{anm}_{date}.mat')
me_file = scipy.io.loadmat(me_path)
me_var = me_file['me']
...
if me_var.dtype.names:
    me_struct = me_var[0, 0]
    me_data = me_struct['data']
    if me_data.dtype.names and 'data' in me_data.dtype.names:
        inner = me_data[0, 0]
        me_raw = inner['data']
...
traj_view0 = obj['traj'][0]
```

iii. The notes justify this by citing the standalone motion-energy files and their nested MATLAB struct variants.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each trial, the AI linearly interpolates the raw per-frame motion-energy trace onto the neural time axis, fills NaNs by nearest valid sample, and then discretizes the resulting aligned trace at the per-session 50th percentile.

ii. 
```python
f_interp = interp1d(old_time, me_trial, kind='linear',
                   bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_interp(time_axis)
...
col[nans] = col[valid_idx[nearest]]
...
me_disc = discretize_per_session(me_valid)
```

iii. The notes say motion energy should be aligned with video offset correction and discretized at the session median.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI uses the same binary `discretize_per_session` rule as for tongue and paw: below the session median is `0`, at or above is `1`, and NaNs become `0`. There is no dedicated "no video" class.

ii. 
```python
def discretize_per_session(data_2d, percentile=50):
    ...
    result = (data_2d >= threshold).astype(np.int32)
    result[np.isnan(data_2d)] = 0
    return result
...
me_disc = discretize_per_session(me_valid)
```

iii. The notes justify the 50th-percentile split but do not add a third category for missing video.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by subtracting the session video offset and the trial go cue from side-camera frame times, then interpolating to the neural `time_axis`.

ii. 
```python
ft = np.array(traj_view0['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
...
f_interp = interp1d(old_time, me_trial, kind='linear',
                   bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_interp(time_axis)
```

iii. The notes cite the reference video-offset computation as the basis for this alignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI generally fills or fabricates rather than dropping data. Missing `stim.enable` becomes all-false; missing motion-energy files become all-zero output; missing frame times fall back to synthetic `np.arange(...)/VIDEO_FR` times with a `0.5` s offset heuristic; missing motion-energy samples are nearest-filled; missing tongue samples are set to zero velocity; missing paw coordinates are nearest-filled before differentiation; and all-zero neural trials at the end of some sessions are retained instead of removed.

ii. 
```python
bp['stim'] = {'enable': np.zeros(int(bp_raw['Ntrials'].flat[0]))}
...
except:
    nframes = len(me_trial)
    ft = np.arange(1, nframes + 1) / VIDEO_FR
    old_time = ft - 0.5 - align_times[trix]
...
col[nans] = col[valid_idx[nearest]]
...
tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)
...
arr[nans] = arr[valid[nearest]]
```

iii. The notes justify some of these choices pragmatically: missing tongue should mean no movement, nearest-fill for paw should stabilize velocity estimation, and all-zero neural trials were kept "for completeness."

## 11-a. What are the most time-consuming steps of the code?

i. The code is instrumented around three phases: session loading, spike binning, and behavioral processing. From the structure of the code, the most expensive computation is the nested spike-binning loop over every kept cluster and every trial, followed by the per-trial interpolation loops for tongue, paw, and motion energy.

ii. 
```python
t0 = time.time()
...
t_load = time.time() - t0
...
t_bin_start = time.time()
for probe_idx, valid_clu in all_cluster_indices:
    ...
    for i, clu_idx in enumerate(valid_clu):
        ...
        for j in range(ntrials_total):
            ...
t_bin = time.time() - t_bin_start
...
t_behav_start = time.time()
tongue_vel = compute_tongue_velocity(...)
paw_vel = compute_paw_velocity(...)
me_aligned = get_motion_energy_aligned(...)
```

iii. The notes record total conversion runtime and emphasize timing information, but they do not give a sharper justification than that.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike-processing code uses nested Python loops over probe, cluster, and trial when it could have been vectorized across spikes and trials. The behavior code also loops over trials for motion energy, tongue, and paw interpolation, and parses feature names separately in multiple functions.

ii. 
```python
for probe_idx, valid_clu in all_cluster_indices:
    for i, clu_idx in enumerate(valid_clu):
        ...
        for j in range(ntrials_total):
            spk_mask = trial_arr == trial_num
            ...

for trix in range(ntrials):
    ...
    f_interp = interp1d(old_time, speed, ...)
```

iii. There is no explicit justification in the notes for leaving these loops unvectorized, beyond a general desire to keep the code understandable and timed.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several operations: it recomputes spike histograms separately for every cluster and every trial; it performs similar frame-time interpolation patterns independently for motion energy, tongue, and paw; it re-parses bottom-camera feature names in both tongue and paw functions; and it defines `align_and_bin_spikes` and `remove_low_fr_clusters` but then reimplements the same logic inside `process_session`.

ii. 
```python
def align_and_bin_spikes(...):
    ...

def remove_low_fr_clusters(...):
    ...

for i, clu_idx in enumerate(valid_clu):
    ...
    for j in range(ntrials_total):
        counts, _ = np.histogram(aligned, bins=edges)
...
ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
```

iii. No explicit justification for this repetition appears in the notes or trajectory.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI's custom v5 loader materializes many fields that the conversion never uses, including `ex`, much of `sglx`, and non-target trajectory fields. It also reads `moveThresh` into `me_thresh` and then never uses it, imports modules such as `glob` and `sys` without using them, allocates unused variables such as `vx` and `vy` in tongue processing, and defines helper functions that are never called.

ii. 
```python
import glob
import sys
...
me_thresh = None
...
return obj, me_raw, me_thresh, t_load
...
vx = np.full_like(x, np.nan)
vy = np.full_like(y, np.nan)
...
def align_and_bin_spikes(...):
    ...
def remove_low_fr_clusters(...):
    ...
```

iii. The notes do not justify these extra reads or unused helpers; they appear to be by-products of building a general loader and multiple draft implementations.
