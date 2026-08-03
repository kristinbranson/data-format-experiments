# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a 44-session manifest in `SESSION_DEFS`, splits those sessions across `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`, and loads each session from `data_structure_<animal>_<date>.mat`. It detects MATLAB v7.3 versus v5 format, reads `bp`, `clu`, and `traj` from the session file, and separately tries to load `motionEnergy_<animal>_<date>.mat`.

ii.
```python
SESSION_DEFS = [
    ('EKH1', '2021-08-07', [2], 'ephys'),
    ...
    ('JEB24', '2023-11-03', [1], 'random'),
]

fmt = _detect_file_format(data_fn)
if fmt == 'hdf5':
    raw = _load_raw_data_hdf5(data_fn, probes)
else:
    raw = _load_raw_data_v5(data_fn, probes)
```

```python
data_fn = os.path.join(data_dir, f'data_structure_{anm}_{date}.mat')
me_fn = os.path.join(data_dir, f'motionEnergy_{anm}_{date}.mat')
```

iii. In `CONVERSION_NOTES.md`, the AI says it transcribed sessions from the loading scripts, excluded files without loading scripts, and used both MATLAB readers because the shared data contain both HDF5 and v5 `.mat` layouts.

## 1-b. How are the data split into subjects?

i. The AI treats the first element of each `SESSION_DEFS` tuple (`anm`) as the subject id, then builds `subjects` as the sorted unique animal names and `subject_idx` as one index per session.

ii.
```python
return {
    'session_id': session_id,
    'animal': anm,
    'date': date,
    ...
}
```

```python
all_animals = sorted(set(s['animal'] for s in all_sessions))
animal_to_idx = {a: i for i, a in enumerate(all_animals)}
...
'subjects': all_animals,
'subject_idx': np.array(subject_idx, dtype=np.int64),
```

iii. The notes justify this by treating the loading-script animal names as authoritative subject ids and by describing the dataset as organized by session files named with animal and date.

## 1-c. How are the data split into sessions?

i. One tuple in `SESSION_DEFS` is one session. Each tuple provides animal, date, probe list, and which top-level folder to read from, and each processed session becomes one element of `neural`, `input`, and `output`.

ii.
```python
for i, (anm, date, probes, ddir) in enumerate(session_defs):
    result = load_session(anm, date, probes, ddir, PARAMS)
    if result is not None:
        all_sessions.append(result)
```

```python
data = {
    'neural': neural,
    'input': inputs,
    'output': outputs,
    ...
}
```

iii. In the notes, the AI says it used all sessions present in the loading scripts and grouped them uniformly across the two task folders.

## 1-d. How are the data split into trials?

i. Trials are defined from the Bpod trial table. The AI reads `Ntrials`, applies a trial mask from `bp['early']` and `bp['stim_enable']`, uses those surviving trial numbers as `valid_trials`, and then indexes neural and behavioral arrays by those trial numbers.

ii.
```python
bp['Ntrials'] = int(bp_group['Ntrials'][()].flat[0])
...
trial_mask = (bp['early'] == 0) & (bp['stim_enable'] == 0)
valid_trials = np.where(trial_mask)[0] + 1  # 1-indexed
```

```python
for t_idx, trial_num in enumerate(valid_trials):
    spk_mask = spike_trial == trial_num
    ...
valid_trial_indices = valid_trials - 1
```

iii. The notes frame this as following the Bpod session structure directly, with one row per behavioral trial and trial numbers carried through spike and video data.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops early-lick and photostimulation trials, requires at least two surviving trials per session, and then keeps later trials even if they have no spikes in the alignment window. It does not implement the reference solution’s extra drop of trials that extend beyond the end of the recording.

ii.
```python
trial_mask = (bp['early'] == 0) & (bp['stim_enable'] == 0)
valid_trials = np.where(trial_mask)[0] + 1  # 1-indexed
if len(valid_trials) < 2:
    print(f"  WARNING: {session_id} has < 2 valid trials, skipping")
    return None
```

iii. The notes explicitly justify excluding early and stim trials “per reference code conditions,” and later document all-zero late trials as an acceptable known issue rather than filtering them out.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural output is derived from spike times and trial assignments in `obj.clu`, plus the trial-wise go-cue times in `obj.bp.ev.goCue` for alignment. Quality labels are also read from `obj.clu` for neuron filtering.

ii.
```python
quality = h5_read_string(f, quality_refs[i]).strip().lower()
trialtm = f[trialtm_refs[i]][()].flatten().astype(np.float64)
trial = f[trial_refs[i]][()].flatten().astype(np.float64)
```

```python
align_times = ev[params['align_event']]
spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
```

iii. The notes describe the intended neural pipeline as “alignSpikes -> getSeq -> removeLowFRClusters,” so the AI’s rationale was that `trialtm`, `trial`, `quality`, and `goCue` are the raw ingredients needed to recreate that pipeline.

## 2-b. How is the `neural` data processed?

i. The AI bins spikes from `-2.5` to `2.5` seconds around go cue in `10 ms` bins, converts counts to firing rates by dividing by `dt`, and applies a custom causal Gaussian smoothing kernel with window length `15`. It performs no additional normalization or baseline subtraction.

ii.
```python
PARAMS = {
    'align_event': 'goCue',
    'tmin': -2.5,
    'tmax': 2.5,
    'dt': 1.0 / 100,  # 10 ms bins
    'smooth_window': 15,
    'smooth_bctype': 'reflect',
    ...
}
```

```python
counts, _ = np.histogram(spk_times, bins=edges)
fr = causal_gaussian_smooth(counts.astype(np.float64) / params['dt'],
                           params['smooth_window'],
                           params['smooth_bctype'])
trialdat[neuron_idx, :, t_idx] = fr.astype(np.float32)
```

iii. The notes say the AI chose `10 ms` bins and causal Gaussian smoothing because it believed `WorkingWithDataObjs.m` was the authoritative tutorial and that `mySmooth.m` should be replicated directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first excludes clusters whose lower-cased quality label is in `{'garbage', 'noisy', 'gabrga', 'real?'}` or whose label is empty/`nan`. It then removes neurons with mean firing rate `<= 1 Hz` across all kept trials and bins, and finally discards any session with fewer than `10` remaining units.

ii.
```python
'quality_exclude': {'garbage', 'noisy', 'gabrga', 'real?'},
'low_fr': 1.0,
'min_units': 10,
```

```python
keep_mask = np.array([q not in quality_exclude and q != '' and q != 'nan'
                      for q in all_qualities])
...
mean_fr = np.mean(trialdat, axis=(1, 2))
fr_mask = mean_fr > params['low_fr']
trialdat = trialdat[fr_mask, :, :]
...
if n_neurons < params['min_units']:
    ...
```

iii. In the notes, the AI justifies these settings as matching the “quality='all'” tutorial mode and the paper’s 1 Hz / 10-unit thresholds.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For each spike, the AI subtracts that trial’s go-cue time from the spike’s within-trial time, then bins the result on the common session time axis. This makes every neural trial relative to go-cue onset.

ii.
```python
align_times = ev[params['align_event']]
...
spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
counts, _ = np.histogram(spk_times, bins=edges)
```

iii. The notes repeatedly state that all streams should be aligned to `goCue`, and the function comment says it is following `alignSpikes`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use `10 ms` bins over a `5 s` window, giving `500` time points per trial. There is no later rebinning; the same `10 ms` grid is used for all streams.

ii.
```python
'dt': 1.0 / 100,  # 10 ms bins
...
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
time_axis = edges[:-1] + params['dt'] / 2
n_timebins = len(time_axis)
```

iii. The AI’s notes explicitly say “10ms time bins” and claim this matches `WorkingWithDataObjs.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The AI does not derive this input from a separate raw time series. It constructs a shared time axis from the alignment window parameters and uses the go-cue-aligned neural grid as the meaning of the variable.

ii.
```python
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
time_axis = edges[:-1] + params['dt'] / 2
```

```python
time_input = time_axis.astype(np.float32).reshape(1, -1)
sess_input.append(time_input)
```

iii. In the notes, the AI says this input is a “linear ramp from -2.5 to 2.5s” and treats it as defined by the chosen go-cue-centered bin grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes the centers of the common time bins and repeats that 1D vector for every trial. No other processing is applied.

ii.
```python
time_axis = edges[:-1] + params['dt'] / 2
...
time_input = time_axis.astype(np.float32).reshape(1, -1)
sess_input.append(time_input)
```

iii. The notes justify this as the simplest input consistent with the decoder prompt: one continuous, time-varying variable measuring time from go cue.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the exact same `time_axis` as the neural binning grid, so each input bin corresponds to the same bin used for spike counting and smoothing.

ii.
```python
time_axis = edges[:-1] + params['dt'] / 2
...
sess_neural.append(trialdat[:, :, t].astype(np.float32))
...
time_input = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes say the time input is meant to be “continuous, time-varying” and aligned by construction because it is copied from the shared go-cue-centered grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction only from the raw trial flag `obj.bp.R`. It does not combine that flag with `hit`, `miss`, or `no` to infer the actual lick that occurred.

ii.
```python
lick_direction = bp['R'][valid_trial_indices].copy()
```

iii. The notes say “obj.bp.R/L” map to lick direction and that this is a per-trial output; there is no separate reasoning in the code or notes about miss or ignore trials.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI performs a direct copy of the `R` indicator for each valid trial and then broadcasts that 0/1 value across all time bins in the trial.

ii.
```python
lick_direction = bp['R'][valid_trial_indices].copy()
...
lick_dir = np.full((1, n_timebins), int(sess['lick_direction'][t]), dtype=np.int64)
```

iii. The notes justify this as “L=0, R=1, per-trial” from trial info, without discussing that the paper’s actual lick direction must be inferred from trial outcome.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`.

ii.
```python
for field in ['hit', 'miss', 'no', 'early', 'L', 'R', 'autowater']:
    bp[field] = bp_group[field][()].flatten().astype(np.float64)[:ntrials]
...
behavioral_context = 1.0 - bp['autowater'][valid_trial_indices]
```

iii. The notes explicitly map `autowater=1` to WC and `autowater=0` to DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI relabels `autowater` into the requested coding by computing `1 - autowater`, which yields `0` for WC and `1` for DR, then broadcasts that per-trial value across time bins.

ii.
```python
behavioral_context = 1.0 - bp['autowater'][valid_trial_indices]
...
context = np.full((1, n_timebins), int(sess['behavioral_context'][t]), dtype=np.int64)
```

iii. The notes say this mapping was chosen to satisfy the decoder prompt’s required coding `WC=0, DR=1`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI derives outcome only from `obj.bp.hit`. It does not use `miss` or `no` to distinguish incorrect from ignored trials.

ii.
```python
outcome = bp['hit'][valid_trial_indices].copy()
```

iii. The notes describe this as “hit=1 -> correct=1, miss=1 -> incorrect=0,” but the code itself only copies `hit`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI uses the `hit` flag directly as a binary correct/incorrect label and broadcasts it across all bins in the trial. Any non-hit trial becomes `0`.

ii.
```python
outcome = bp['hit'][valid_trial_indices].copy()
...
outc = np.full((1, n_timebins), int(sess['outcome'][t]), dtype=np.int64)
```

iii. The notes justify this as matching the prompt’s requested binary coding for incorrect versus correct, even though the reference solution keeps ignore trials as a third class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI derives tongue velocity from DeepLabCut trajectories in `obj.traj`, specifically from the bottom-camera feature `top_tongue`, together with `frameTimes`, the video shift estimate, and go-cue times.

ii.
```python
tongue_vel = _compute_feature_velocity_generic(
    traj_data, ntrials, view=2, feat_name='top_tongue',
    vidshift=vidshift, align_times=align_times,
    time_axis=time_axis, is_tongue=True)
```

iii. In the notes, the AI explicitly chose “bottom cam DLC features (top_tongue or similar)” for tongue velocity.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI interpolates `x` and `y` positions from the bottom camera directly onto the neural time axis, takes simple gradients of those interpolated traces, sets NaN tongue gradients to zero, and computes speed as `sqrt(xvel^2 + yvel^2)`. It does not threshold on DLC likelihood, smooth within contiguous tracked runs, normalize separate camera views, or average side and bottom views.

ii.
```python
xpos = interp1d(ft_a[valid], x_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
ypos = interp1d(ft_a[valid], y_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
...
xvel = np.gradient(xpos)
yvel = np.gradient(ypos)
...
xvel[np.isnan(xvel)] = 0
yvel[np.isnan(yvel)] = 0
speed[:, trix] = np.sqrt(xvel**2 + yvel**2).astype(np.float32)
```

iii. The notes justify this choice only at a high level: use Euclidean velocity from bottom-camera tongue tracking and discretize later for the decoder.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI computes the session-wide 50th percentile across all non-NaN tongue-velocity samples, labels values above or equal to the threshold as `1`, below as `0`, and converts NaN samples to `0`.

ii.
```python
def _discretize_velocity(data, n_timebins, n_trials):
    ...
    valid_vals = data[~np.isnan(data)]
    ...
    threshold = np.percentile(valid_vals, 50)
    disc = (data >= threshold).astype(np.int64)
    disc[np.isnan(data)] = 0
    return disc
```

iii. The notes explicitly say “Per-session discretization: 50th percentile threshold computed on all timepoints across all trials in session.”

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. For each trial, the AI subtracts the video shift and trial go-cue time from `frameTimes`, interpolates the tongue position to the neural `time_axis`, and then computes velocity on that neural grid. Alignment therefore happens by interpolation onto the same per-trial time vector used for spikes.

ii.
```python
ft_aligned = ft - vidshift - align_times[trix]
...
xpos = interp1d(ft_a[valid], x_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
```

iii. The notes say the AI followed the reference idea of subtracting the video offset / bit-start mismatch and aligning to go cue, but implemented the final step as interpolation rather than frame binning.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from `obj.traj`, using the bottom-camera feature `top_paw`, along with `frameTimes`, video shift, and go-cue times.

ii.
```python
paw_vel = _compute_feature_velocity_generic(
    traj_data, ntrials, view=2, feat_name='top_paw',
    vidshift=vidshift, align_times=align_times,
    time_axis=time_axis, is_tongue=False)
```

iii. The notes explicitly say the AI chose the bottom-camera paw features and intended to use Euclidean velocity.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI interpolates paw position onto the neural time axis, fills missing positions by nearest-neighbor carry-forward/backward, takes gradients, subtracts the median first-difference baseline from each velocity component, fills NaN velocities by nearest values, and computes Euclidean speed. It does not use the reference pipeline’s DLC likelihood thresholding, within-run smoothing, or frame-bin averaging.

ii.
```python
xpos = interp1d(ft_a[valid], x_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
ypos = interp1d(ft_a[valid], y_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)

xpos = _fill_nearest(xpos)
ypos = _fill_nearest(ypos)
...
xvel = np.gradient(xpos)
yvel = np.gradient(ypos)
base_xvel = np.nanmedian(np.diff(xpos))
base_yvel = np.nanmedian(np.diff(ypos))
...
speed[:, trix] = np.sqrt(xvel**2 + yvel**2).astype(np.float32)
```

iii. The notes justify this only generally as deriving paw velocity from bottom-camera DLC trajectories and discretizing it later.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI uses the same helper as tongue velocity: session-wide median split across non-NaN paw-speed values, with NaNs reassigned to class `0`.

ii.
```python
paw_vel_disc = _discretize_velocity(paw_vel, n_timebins, n_valid_trials)
...
threshold = np.percentile(valid_vals, 50)
disc = (data >= threshold).astype(np.int64)
disc[np.isnan(data)] = 0
```

iii. The notes say all continuous movement outputs are discretized per session at the 50th percentile.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The AI aligns paw tracking by subtracting video shift and trial go-cue time from frame times, then interpolating paw position to the same `time_axis` used for neural data before taking derivatives.

ii.
```python
ft_aligned = ft - vidshift - align_times[trix]
...
xpos = interp1d(ft_a[valid], x_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
```

iii. The notes indicate the intended alignment was “interpolate to neural time axis,” which the AI considered consistent with the motion-energy alignment approach it adopted.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<animal>_<date>.mat` file for each session, plus side-camera frame times from `traj_data` when available, plus the same video shift and go-cue alignment variables.

ii.
```python
me_fn = os.path.join(data_dir, f'motionEnergy_{anm}_{date}.mat')
...
me_data = _load_motion_energy_generic(me_fn, traj_data, vidshift,
                                      align_times, time_axis, ntrials)
```

```python
side_cam = traj_data['views'][0] if traj_data is not None else None
...
if side_cam is not None and trix < len(side_cam['trials']) and side_cam['trials'][trix] is not None:
    ft = side_cam['trials'][trix]['frameTimes']
```

iii. The notes justify this as following `loadMotionEnergy` and using the paired motion-energy files rather than another derived signal.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI linearly interpolates each trial’s motion-energy trace from frame times onto the neural `time_axis`, then fills any interior NaNs with nearest values. It does not re-compute motion energy from pixels, and it does not use the reference solution’s frame-bin averaging with NaNs left as missing.

ii.
```python
me_aligned[:, trix] = interp1d(ft_aligned[valid], me_trial[valid],
                               kind='linear', bounds_error=False,
                               fill_value=np.nan)(time_axis).astype(np.float32)
...
if not np.isnan(col).all() and np.isnan(col).any():
    me_aligned[:, trix] = _fill_nearest(col)
```

iii. The notes say “Motion energy: load from motionEnergy files, interpolate to neural time axis,” so the AI’s rationale was that interpolation was the appropriate alignment step.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI uses the same `_discretize_velocity` helper here too: session-wide 50th percentile across all finite motion-energy samples, class `1` for values above or equal to threshold, class `0` below threshold, and NaNs reassigned to `0`.

ii.
```python
me_disc = _discretize_velocity(me_data, n_timebins, n_valid_trials)
...
threshold = np.percentile(valid_vals, 50)
disc = (data >= threshold).astype(np.int64)
disc[np.isnan(data)] = 0
```

iii. The notes state that motion energy, like the velocity outputs, should use a per-session 50th-percentile discretization.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy by subtracting video shift and trial go-cue time from side-camera frame times and interpolating the motion-energy trace onto the shared neural `time_axis`.

ii.
```python
ft_aligned = ft - vidshift - align_times[trix]
...
me_aligned[:, trix] = interp1d(ft_aligned[valid], me_trial[valid],
                               kind='linear', bounds_error=False,
                               fill_value=np.nan)(time_axis).astype(np.float32)
```

iii. The notes explicitly describe this as interpolation to the neural axis after go-cue / video-offset correction.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI tends to impute or collapse missing data rather than preserve an explicit missing/visibility state. Missing frame times are replaced with synthetic `1/400 s` timestamps; missing paw positions and partially missing motion-energy traces are nearest-filled; tongue NaNs are converted to zero velocity; fully missing continuous outputs become all-zero discrete outputs. Corrupted motion-energy files are tolerated by leaving `me_data` as `None`, which later becomes all-zero discrete motion energy.

ii.
```python
if ft is None or ft.size == 0 or np.all(np.isnan(ft)):
    ft = np.arange(1, me_trial.size + 1) / 400.0
```

```python
if ft.size == 0 or np.all(np.isnan(ft)):
    ft = np.arange(1, len(x_raw) + 1) / 400.0
...
xpos = _fill_nearest(xpos)
ypos = _fill_nearest(ypos)
...
disc[np.isnan(data)] = 0
```

iii. The notes justify this as “handle missing data appropriately” and later document degenerate motion-energy sessions and all-zero outputs as acceptable known issues rather than filtering or adding a missing-data class.

## 11-a. What are the most time-consuming steps of the code?

i. The AI instrumented conversion at the session level and treated full-session loading / processing as the dominant runtime cost. In the code, the heaviest work is loading each `.mat` session plus the nested neuron-by-trial spike binning and the per-trial interpolation loops for movement and motion energy.

ii.
```python
t0 = time.time()
...
for neuron_idx, clu_idx in enumerate(good_indices):
    ...
    for t_idx, trial_num in enumerate(valid_trials):
        ...
```

```python
for trix in range(ntrials):
    ...
for trix in range(min(ntrials, len(view_data['trials']))):
    ...
print(f"  {session_id}: ... ({elapsed:.1f}s)")
```

iii. The notes estimate about `5 s` per session and about `3.7 min` total for full loading, so the AI’s explicit rationale emphasized end-to-end per-session conversion time rather than micro-optimizing individual kernels.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI left several expensive Python loops in place: the neuron-by-trial spike-binning loop, the per-trial motion-energy interpolation loop, the per-trial feature-velocity loop, and the final per-trial assembly loop. These are all plausible vectorization targets.

ii.
```python
for neuron_idx, clu_idx in enumerate(good_indices):
    ...
    for t_idx, trial_num in enumerate(valid_trials):
        ...
```

```python
for trix in range(ntrials):
    ...
for trix in range(min(ntrials, len(view_data['trials']))):
    ...
for t in range(n_trials):
    ...
```

iii. The notes mention efficiency and runtime estimation, but do not give a detailed vectorization rationale; this section is therefore inferred mainly from the structure of the AI’s code.

## 11-c. What processing does the code repeat multiple times?

i. The AI repeats work at several levels: it rebuilds `time_input` for every trial during assembly, bins spikes separately for every neuron/trial pair instead of batching across trials, and interpolates each movement stream trial-by-trial. It also post-processes motion-energy columns in a second pass to fill NaNs.

ii.
```python
for neuron_idx, clu_idx in enumerate(good_indices):
    ...
    for t_idx, trial_num in enumerate(valid_trials):
        counts, _ = np.histogram(spk_times, bins=edges)
        fr = causal_gaussian_smooth(...)
```

```python
for t in range(n_trials):
    time_input = time_axis.astype(np.float32).reshape(1, -1)
    sess_input.append(time_input)
```

```python
for trix in range(ntrials):
    ...
for trix in range(ntrials):
    col = me_aligned[:, trix]
    ...
```

iii. There is no explicit note defending these repetitions; they appear to be implementation choices made for simplicity.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI reads and stores more intermediate information than the final dataset uses. It loads full trajectory structures for all tracked features and both cameras even though only `top_tongue`, `top_paw`, and side-camera frame times are later used; it computes and stores raw continuous `tongue_vel_raw`, `paw_vel_raw`, and `me_raw` in each session record even though the final pickle only keeps discretized outputs; and it keeps plotting-oriented fields such as `time_axis` and raw traces in memory until final assembly.

ii.
```python
traj_data = {'views': []}
...
feat_names = [h5_read_string(f, fr) for fr in feat_refs_trial]
...
view_trials.append({'ts': ts, 'frameTimes': ft, 'ts_format': 'hdf5'})
```

```python
return {
    ...
    'time_axis': time_axis,
    ...
    'tongue_vel_raw': tongue_vel,
    'paw_vel_raw': paw_vel,
    'me_raw': me_data,
}
```

iii. The notes justify these extras indirectly through `--show-processing` plots and debugging, but they are not retained in `converted_data.pkl` and are therefore discarded after assembly.
