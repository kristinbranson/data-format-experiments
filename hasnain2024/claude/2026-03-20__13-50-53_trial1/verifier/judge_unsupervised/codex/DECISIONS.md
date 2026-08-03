# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes a session registry, then loads one MATLAB `data_structure_<animal>_<date>.mat` file and one matching `motionEnergy_<animal>_<date>.mat` file per listed session. It combines `EPHYS_SESSIONS` and `RANDOMIZED_DELAY_SESSIONS` into `ALL_SESSIONS`, and iterates through that list in `main()`.

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
    ...
    me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{anm}_{date}.mat')
```

```python
if args.sample:
    sessions = ALL_SESSIONS[:2]
else:
    sessions = ALL_SESSIONS

for i, (anm, date, probes, data_dir) in enumerate(sessions):
    result = process_session(anm, date, probes, data_dir, ...)
```

iii. `CONVERSION_NOTES.md` says the registry was taken from the paper’s loading scripts and that the key decision was to “Include all 44 ephys sessions.” The trajectory also records a deliberate choice to process all 25 `Ephys_Behavior` sessions plus 19 `RandomizedDelay_Ephys_Behavior` sessions.

## 1-b. How are the data split into subjects?

i. Sessions are assigned to subjects by the animal ID string in each session tuple. The code collects unique animal names, sorts them, and stores a per-session `subject_idx`.

ii. ```python
all_animals.append(anm)
subjects_set.add(anm)
...
subjects = sorted(subjects_set)
subject_idx = np.array([subjects.index(anm) for anm in all_animals], dtype=np.int64)
```

iii. The notes describe subject IDs as the animal names from the loading scripts, and the full-conversion summary lists 14 subject names derived this way.

## 1-c. How are the data split into sessions?

i. Each tuple in `ALL_SESSIONS` defines one session. `process_session()` returns one session-level bundle of `neural`, `input`, and `output` trial lists, which `main()` appends as one session entry. Sessions are skipped only if they have fewer than 2 valid trials or fewer than 10 neurons after filtering.

ii. ```python
for i, (anm, date, probes, data_dir) in enumerate(sessions):
    result = process_session(anm, date, probes, data_dir, ...)
    if result is None:
        continue
    all_neural.append(result['neural'])
    all_input.append(result['input'])
    all_output.append(result['output'])
```

```python
if n_valid < 2:
    return None
...
if n_neurons_final < 10:
    return None
```

iii. The notes explicitly say “Sessions to Include (from loading scripts)” and list the hard-coded session table that the Python script later mirrors.

## 1-d. How are the data split into trials?

i. Trials are taken from the per-session Bpod arrays in `obj['bp']`. The script builds a Boolean `valid_mask`, converts it to `valid_trials`, then iterates over valid trial indices to create one neural matrix, one input array, and one output array per trial.

ii. ```python
ntrials_total = int(bp['Ntrials'])
...
valid_trials = np.where(valid_mask)[0]
...
trialdat_valid = trialdat[:, :, valid_trials]
```

```python
for t_idx in range(n_valid):
    neural_trials.append(trialdat_valid[:, :, t_idx].T.astype(np.float32))
    input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
    ...
    output_trials.append(out)
```

iii. The notes describe the target as session lists containing trial lists, and the trajectory’s verification steps refer to exact counts of valid trials per session computed from this mask.

## 1-e. How are trials filtered based on quality controls?

i. The agent excludes early-lick trials, optogenetic stimulation trials, and no-response trials, and also requires `hit | miss`. It does not implement the paper’s session-level behavioral inclusion criteria, and it keeps later artifact trials with all-zero neural data.

ii. ```python
stim_enable = np.zeros(ntrials_total, dtype=bool)
...
valid_mask = ~early & ~stim_enable & ~no
valid_mask = valid_mask & (hit | miss)
valid_trials = np.where(valid_mask)[0]
```

iii. `CONVERSION_NOTES.md` states “Trial filtering (exclude early, stim.enable, no-response)” and a later verification step checks this logic on one session. The same notes separately acknowledge paper-level behavioral inclusion rules, but the code never enforces them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from spike cluster data in `obj['clu']`, especially the per-spike trial labels, per-spike trial times, and cluster quality labels; alignment uses `bp.ev.goCue`.

ii. ```python
goCue = np.array(ev['goCue']).flatten()
...
trial_arr = np.array(clu_probe['trial'][clu_idx]).flatten()
trialtm_arr = np.array(clu_probe['trialtm'][clu_idx]).flatten()
...
qualities = clu_probe['quality']
```

iii. The notes map `obj.clu` to the `neural` field and identify `alignSpikes`, `getSeq`, and `removeLowFRClusters` as the reference functions behind that mapping.

## 2-b. How is the `neural` data processed?

i. For each kept cluster and trial, the code aligns spike times to go cue, bins spikes from -2.5 s to 2.5 s in 10 ms bins, converts counts to firing rates, and applies a causal Gaussian smoother with a 15-bin window and reflect padding.

ii. ```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
...
aligned = trialtm_arr[spk_mask] - align_times_all[j]
counts, _ = np.histogram(aligned, bins=edges)
fr = counts.astype(np.float32) / DT
trialdat[:, neuron_offset + i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. The notes say this was chosen to match `alignSpikes.m`, `getSeq.m`, and `mySmooth.m`, and the trajectory records an exact raw-versus-converted spike-trace spot check on EKH1.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script first removes clusters whose quality label is one of `garbage`, `gabrga`, `noisy`, or `real?`, then removes neurons whose mean firing rate across all times and trials is not greater than 1 Hz. Entire sessions with fewer than 10 surviving neurons are dropped.

ii. ```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
...
if q_stripped.lower() not in {e.lower() for e in excluded_qualities}:
    valid.append(i)
```

```python
mean_fr = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
keep = mean_fr > low_fr
return trialdat[:, keep, :], keep
```

```python
if n_neurons_final < 10:
    return None
```

iii. The notes cite `findClusters.m` and `removeLowFRClusters.m`, and say the agent used the paper’s 1 Hz threshold rather than lower defaults seen elsewhere in the repo.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to go cue onset by subtracting the trial’s `goCue` time from each spike’s trial-relative timestamp before binning.

ii. ```python
ALIGN_EVENT = 'goCue'
...
align_times_all = goCue
...
aligned = trialtm_arr[spk_mask] - align_times_all[j]
```

iii. The notes repeatedly state that the reference alignment event is `goCue`, and the trajectory’s spike-verification step checks the resulting alignment explicitly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins (`DT = 1/100`) over a fixed 5 s window around go cue. No later rebinning is applied.

ii. ```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 100
...
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
```

iii. The notes say the agent chose 10 ms because `WorkingWithDataObjs.m` uses `dt = 1/100`, even though some other code paths use 5 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This input is not derived from a raw time series in the `.mat` file. It is derived from the chosen alignment convention: go cue defines time zero, and the actual values come from `TMIN`, `TMAX`, and `DT`.

ii. ```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 100
...
goCue = np.array(ev['goCue']).flatten()
...
time_axis = edges[:-1] + DT / 2
```

iii. The notes map the decoder input to “time from goCue (s)” and describe it as a continuous axis ranging from -2.5 to 2.5 s.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code builds bin edges, converts them to bin centers, and stores the same 1 x T time axis for every valid trial in a session.

ii. ```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
...
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. The notes justify this as using the same time base as the reference neural binning pipeline.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the exact same `time_axis` array used to bin neural spikes, so every input bin is aligned one-for-one with every neural time bin.

ii. ```python
trialdat = np.zeros((n_time, total_neurons, ntrials_total), dtype=np.float32)
...
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. The notes and verification steps both describe the time axis as matching the neural bin centers exactly.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the Bpod task arrays `R`, `L`, `hit`, and `miss`.

ii. ```python
R = np.array(bp['R']).flatten().astype(bool)
L = np.array(bp['L']).flatten().astype(bool)
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
```

iii. The notes map `obj.bp.R/L + hit/miss` to `output[0]`.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code labels right licks as `(R & hit) | (L & miss)` and left licks as the complement implied by misses/hits on the opposite side, then broadcasts the per-trial value across all time bins.

ii. ```python
lick_right = (R & hit) | (L & miss)
lick_direction = lick_right[valid_trials].astype(np.int32)
...
out[0, :] = lick_direction[t_idx]
```

iii. The notes and a trajectory verification step both describe and check this exact mapping.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the Bpod `autowater` field.

ii. ```python
autowater = np.array(bp['autowater']).flatten().astype(bool)
```

iii. `WorkingWithDataObjs.m` was read in the trajectory and explicitly described `obj.bp.autowater` as a proxy for WC versus DR blocks; the notes repeat that mapping.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The script maps `autowater = 1` to WC (`0`) and `autowater = 0` to DR (`1`), then broadcasts that session-trial label across time bins.

ii. ```python
context = (~autowater[valid_trials]).astype(np.int32)
...
out[1, :] = context[t_idx]
```

iii. The notes state this choice directly: “autowater=1 -> WC(0); autowater=0 -> DR(1).”

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `hit` and `miss` trial outcome arrays in `obj.bp`.

ii. ```python
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
```

iii. The notes map `obj.bp.hit/miss` to `output[2]`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code encodes `hit` as correct (`1`) and, after trial filtering, everything else kept as incorrect (`0`), then broadcasts that value across all time bins in the trial output array.

ii. ```python
outcome = hit[valid_trials].astype(np.int32)
...
out[2, :] = outcome[t_idx]
```

iii. The notes explicitly document “hit -> correct(1); miss -> incorrect(0).”

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The agent derives tongue velocity from DeepLabCut bottom-camera trajectory data: `traj[1]['featNames']`, `traj[1]['ts']`, and `traj[1]['frameTimes']`. It looks for a feature named `top_tongue`, or any feature containing “tongue”.

ii. ```python
traj_bottom = obj['traj'][1]
feat_names_raw = traj_bottom['featNames'][0]
...
for i, name in enumerate(feat_names):
    if name == 'top_tongue':
        tongue_idx = i
        break
```

```python
ts = np.array(traj_bottom['ts'][trix])
...
x = ts[:, 0, tongue_idx].copy()
y = ts[:, 1, tongue_idx].copy()
```

iii. The notes say tongue velocity was taken from DLC and that the agent decided to use “tip-of-tongue displacement from bottom cam view.”

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code computes x/y gradients at 400 Hz on the selected tongue feature, converts them to speed magnitude, interpolates that speed to the neural time axis, and finally replaces all missing values with zero.

ii. ```python
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

iii. The notes justify the no-fill-before-velocity choice from the methods (“tongue missing values NOT filled”), while trajectory steps 95/98/101 show the agent added the final NaN-to-zero step after discovering that tongue visibility was sparse.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. It is thresholded with `discretize_per_session()`, which uses the session-wide 50th percentile over all valid bins, except when that percentile is exactly zero: then it replaces the threshold with floating-point epsilon so zero values become “low”.

ii. ```python
threshold = np.percentile(all_vals, percentile)
if threshold == 0:
    threshold = np.finfo(np.float32).eps
result = (data_2d >= threshold).astype(np.int32)
result[np.isnan(data_2d)] = 0
```

iii. The trajectory documents that the agent added this after seeing tongue velocity become 100% “high” when the median was zero.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. For each trial, the code builds video time relative to go cue by subtracting the video-neural offset and the trial’s go cue time, then linearly interpolates tongue speed onto the neural `time_axis`.

ii. ```python
ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
...
f_interp = interp1d(old_time, speed, kind='linear',
                   bounds_error=False, fill_value=np.nan)
tongue_vel[:, trix] = f_interp(time_axis)
```

iii. The notes say the agent matched the video-alignment logic used elsewhere in the reference pipeline, using a per-session offset from bitcode timing.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera DLC data. The script scans `traj[1]['featNames']` for any feature containing “paw”, then uses those features’ x/y coordinates from `traj[1]['ts']` and frame times from `traj[1]['frameTimes']`.

ii. ```python
traj_bottom = obj['traj'][1]
...
paw_indices = []
for i, name in enumerate(feat_names):
    if 'paw' in name.lower():
        paw_indices.append(i)
```

```python
ts = np.array(traj_bottom['ts'][trix])
...
x = ts[:, 0, pidx].copy()
y = ts[:, 1, pidx].copy()
```

iii. The notes say paw velocity was computed from DLC using bottom-camera paw features.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature, missing x/y coordinates are filled by nearest neighbor, speed magnitude is computed from 400 Hz gradients, speeds are averaged across paw features, then interpolated to the neural time base.

ii. ```python
for arr in [x, y]:
    nans = np.isnan(arr)
    if nans.any() and not nans.all():
        valid = np.where(~nans)[0]
        nan_pos = np.where(nans)[0]
        nearest = np.searchsorted(valid, nan_pos).clip(0, len(valid)-1)
        arr[nans] = arr[valid[nearest]]
```

```python
vx = np.gradient(x) * VIDEO_FR
vy = np.gradient(y) * VIDEO_FR
speeds.append(np.sqrt(vx**2 + vy**2))
avg_speed = np.mean(speeds, axis=0)
```

iii. The notes justify nearest-neighbor filling for non-tongue kinematic features based on the methods summary.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity uses the same `discretize_per_session()` routine as tongue velocity and motion energy: a session-wide 50th percentile threshold over all bins and valid trials, with a zero-threshold epsilon workaround if needed.

ii. ```python
paw_vel_valid = paw_vel[:, valid_trials]
paw_vel_disc = discretize_per_session(paw_vel_valid)
```

iii. The notes say continuous outputs were discretized at the per-session 50th percentile.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity uses the same per-trial alignment as tongue velocity: frame times are shifted by `vidshift` and the trial’s go cue, then linearly interpolated to the neural bin centers.

ii. ```python
ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
...
f_interp = interp1d(old_time, avg_speed, kind='linear',
                   bounds_error=False, fill_value=np.nan)
paw_vel[:, trix] = f_interp(time_axis)
```

iii. The notes describe this as matching the reference video/neural alignment strategy.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from the separate session-level `motionEnergy_<animal>_<date>.mat` file, together with video frame times from `obj['traj'][0]['frameTimes']` and bitcode timing fields used to compute `vidshift`.

ii. ```python
me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{anm}_{date}.mat')
...
me_raw = me_data
me_thresh = float(me_struct['moveThresh'].flat[0])
```

```python
def compute_video_offset(obj):
    bitStart = scipy_stats.mode(np.array(obj['bp']['ev']['bitStart']).flatten(), keepdims=False).mode
    bc_bitstart = np.array(obj['sglx']['bitcode']['bitstart']).flatten()
```

iii. The notes identify `loadMotionEnergy.m` and the video-offset computation as the relevant reference processing.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code loads the motion-energy trace per trial, builds trial-relative video time, linearly interpolates the trace to the neural time axis, and fills remaining NaN gaps by nearest neighbor.

ii. ```python
me_trial = np.array(me_raw[trix, 0]).flatten()
...
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

iii. The notes say this was intended to match `loadMotionEnergy.m`, including interpolation and nearest-neighbor filling.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The aligned motion-energy values are discretized with the same session-wise percentile function used for the velocity outputs. The raw `moveThresh` value is loaded but not used.

ii. ```python
if me_raw is not None:
    me_aligned = get_motion_energy_aligned(...)
    me_valid = me_aligned[:, valid_trials]
    me_disc = discretize_per_session(me_valid)
```

iii. The notes explicitly say motion energy was discretized at the per-session 50th percentile, even though the raw file contains `me.moveThresh`.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. For each trial, motion energy is aligned by subtracting the session video offset and that trial’s go cue time from the frame times, then interpolating to the neural `time_axis`.

ii. ```python
old_time = ft - vidshift - align_times[trix]
...
me_aligned[:, trix] = f_interp(time_axis)
```

iii. The trajectory includes an independent review reporting a bit-for-bit match between stored motion-energy output and a raw-data recomputation using this alignment rule.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script adds many fallbacks: `mat73` first then a custom SciPy v5 loader; motion-energy files can be nested structs or direct cell arrays; missing tongue data survive until after interpolation and then become zero; missing paw and motion-energy samples are filled by nearest neighbor; absent motion-energy files produce all-zero outputs; sessions with too few valid trials or neurons are skipped; all-zero late-session neural trials are retained.

ii. ```python
try:
    obj = mat73.loadmat(data_path)['obj']
except TypeError:
    obj = _load_v5_session(data_path)
```

```python
if me_var.dtype.names:
    ...
elif me_var.dtype == object:
    me_raw = me_var
    me_thresh = None
```

```python
if tongue_idx is None:
    return np.full((len(time_axis), ntrials), np.nan, dtype=np.float32)
...
tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)
```

```python
else:
    me_disc = np.zeros((n_time, n_valid), dtype=np.int32)
```

iii. The notes and trajectory mention several such fixes explicitly, especially the v5 MATLAB fallback, nested motion-energy format handling, and the tongue-visibility workaround.

## 11-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading large `.mat` files, the nested neuron-by-trial spike-binning loops, and the per-trial interpolation of behavioral signals. The code even prints elapsed time separately for loading, spike binning, and behavioral processing.

ii. ```python
t_start = time.time()
obj, me_raw, me_thresh, t_load = load_session_data(...)
print(f"  Loaded in {t_load:.1f}s")
...
t_bin_start = time.time()
for probe_idx, valid_clu in all_cluster_indices:
    for i, clu_idx in enumerate(valid_clu):
        ...
        for j in range(ntrials_total):
            ...
print(f"  Spike binning: {t_bin:.1f}s for {total_neurons} neurons")
...
print(f"  Behavioral processing: {t_behav:.1f}s")
```

iii. The full-conversion notes say all 44 sessions took about 300 seconds, and the code structure makes spike binning the obvious dominant kernel.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The biggest vectorization targets are the triple nested spike loop (probe -> cluster -> trial), the repeated per-trial interpolation loops for tongue, paw, and motion energy, and the per-column NaN fill loops.

ii. ```python
for probe_idx, valid_clu in all_cluster_indices:
    for i, clu_idx in enumerate(valid_clu):
        ...
        for j in range(ntrials_total):
            ...
```

```python
for trix in range(ntrials):
    ...
    f_interp = interp1d(old_time, ..., kind='linear', ...)
    ...
for trix in range(ntrials):
    col = me_aligned[:, trix]
    ...
```

iii. The notes do not claim these loops were optimized; the trajectory instead emphasizes correctness checks, while the code leaves the heavy loops scalarized.

## 11-c. What processing does the code repeat multiple times?

i. The script repeats the same frame-time alignment and interpolation pattern in the tongue, paw, and motion-energy functions; it parses DLC feature names separately in tongue and paw functions; and it contains helper functions for spike binning and low-FR filtering that are duplicated by the inline logic in `process_session()`.

ii. ```python
old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, ..., kind='linear',
                   bounds_error=False, fill_value=np.nan)
```

```python
def align_and_bin_spikes(...):
    ...

def remove_low_fr_clusters(...):
    ...
```

```python
for fn in feat_names_raw:
    if isinstance(fn, list):
        feat_names.append(fn[0] if fn else '')
    else:
        feat_names.append(str(fn))
```

iii. There is no explicit justification in the notes beyond getting the pipeline working; the repeated logic is visible directly in `convert_data.py`.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and stores `me_thresh` but never uses it; it computes continuous tongue, paw, and motion-energy traces only to immediately binarize them; it loads more raw cluster/session fields than the decoder uses; and it retains raw-trial computations for invalid trials until after neural and behavioral processing, then discards them by slicing `valid_trials`.

ii. ```python
me_raw = None
me_thresh = None
...
return obj, me_raw, me_thresh, t_load
```

```python
tongue_vel = compute_tongue_velocity(...)
tongue_vel_valid = tongue_vel[:, valid_trials]
tongue_vel_disc = discretize_per_session(tongue_vel_valid)
```

```python
trialdat = np.zeros((n_time, total_neurons, ntrials_total), dtype=np.float32)
...
trialdat_valid = trialdat[:, :, valid_trials]
```

iii. The notes focus on correctness and validation, not efficiency. The discarded work is evident from the code path itself.
