# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-coded the session list in `SESSION_DEFS`, with animal, date, probe list, and a flag choosing either `Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`. For each session, `load_session` constructs the `data_structure_<anm>_<date>.mat` and `motionEnergy_<anm>_<date>.mat` paths, detects whether the main file is MATLAB v7.3/HDF5 or v5, and then loads behavioral fields, cluster fields, trajectory fields, and later motion energy.

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

iii. In `CONVERSION_NOTES.md`, the AI says it used the session loading scripts as the source of truth for which sessions to include, excluded extra files without loading scripts, and supported both MATLAB formats because the data include both.

## 1-b. How are the data split into subjects (mice)?

i. The AI treats the `anm` entry in each session tuple as the subject id. During assembly it takes the sorted unique animal names and maps each session to `subject_idx`.

ii.
```python
return {
    'session_id': session_id,
    'animal': anm,
    ...
}

all_animals = sorted(set(s['animal'] for s in all_sessions))
animal_to_idx = {a: i for i, a in enumerate(all_animals)}
subject_idx.append(animal_to_idx[sess['animal']])
```

iii. The notes justify subject assignment from filenames/session definitions rather than relying on inconsistent internal metadata.

## 1-c. How are the data split into sessions?

i. One entry in `SESSION_DEFS` is one session. Each session corresponds to one `data_structure_...mat` file plus its paired motion-energy file and becomes one element of `neural`, `input`, and `output`.

ii.
```python
for i, (anm, date, probes, ddir) in enumerate(session_defs):
    result = load_session(anm, date, probes, ddir, PARAMS)
    if result is not None:
        all_sessions.append(result)
```

iii. The AI's notes say it intentionally unified fixed-delay and randomized-delay sessions through the same loader, following the loading scripts.

## 1-d. How are the data split into trials?

i. The AI uses `bp['Ntrials']` as the session trial count, truncates loaded trial-wise arrays to that count, and represents trials by 1-indexed trial ids in `valid_trials`. Neural spikes are assigned to trials through the raw `trial` field; camera and motion-energy data are read from per-trial trajectory entries.

ii.
```python
bp['Ntrials'] = int(bp_group['Ntrials'][()].flat[0])
...
valid_trials = np.where(trial_mask)[0] + 1  # 1-indexed
...
spike_trial = all_spike_trials[clu_idx]
...
for trix in range(min(ntrials, len(view_data['trials']))):
```

iii. In the notes, the AI describes `obj.bp`, `obj.clu`, and `obj.traj` as already organized per trial, so trial boundaries do not need to be reconstructed.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops trials with `early == 1` or `stim_enable == 1`. It also skips entire sessions with fewer than 2 remaining trials, but it does not implement the reference's extra cut for trials extending past the end of neural recording.

ii.
```python
trial_mask = (bp['early'] == 0) & (bp['stim_enable'] == 0)
valid_trials = np.where(trial_mask)[0] + 1
if len(valid_trials) < 2:
    return None
```

iii. `CONVERSION_NOTES.md` says the AI chose to exclude early-lick and photostimulation trials "per reference code conditions (~early, ~stim.enable)" and to keep all other trial outcomes for decoder targets.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from cluster spike times relative to trial start (`trialtm`), the per-spike trial ids (`trial`), cluster quality labels (`quality`), and the behavioral go-cue times `bp['ev']['goCue']`.

ii.
```python
quality = h5_read_string(f, quality_refs[i]).strip().lower()
trialtm = f[trialtm_refs[i]][()].flatten().astype(np.float64)
trial = f[trial_refs[i]][()].flatten().astype(np.float64)
...
align_times = ev[params['align_event']]
```

iii. The notes summarize the intended neural pipeline as `loadObjs -> findClusters -> alignSpikes -> getSeq -> removeLowFRClusters`, so the AI viewed these raw spike and event fields as the minimal ingredients.

## 2-b. How is the `neural` data processed?

i. The AI aligns each neuron's spike times to go cue by subtraction, bins spikes into 10 ms bins from -2.5 s to 2.5 s, converts counts to firing rates by dividing by `dt`, and smooths each per-trial firing-rate trace with its own causal Gaussian function using a 15-sample window and reflect padding. Units from multiple probes are concatenated.

ii.
```python
PARAMS = {
    'tmin': -2.5,
    'tmax': 2.5,
    'dt': 1.0 / 100,  # 10 ms bins
    'smooth_window': 15,
    'smooth_bctype': 'reflect',
}
...
spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
counts, _ = np.histogram(spk_times, bins=edges)
fr = causal_gaussian_smooth(counts.astype(np.float64) / params['dt'],
                           params['smooth_window'],
                           params['smooth_bctype'])
```

iii. The AI explicitly justified this in the notes: it chose 10 ms bins, a causal Gaussian of window 15, and 1 Hz minimum firing rate because `WorkingWithDataObjs.m` looked like the best reference/tutorial.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps clusters whose lower-cased quality label is not in `{'garbage', 'noisy', 'gabrga', 'real?'}` and is not blank/`nan`. It then removes neurons whose mean firing rate across all bins and kept trials is `<= 1 Hz`. It also drops whole sessions with fewer than 10 remaining units.

ii.
```python
'quality_exclude': {'garbage', 'noisy', 'gabrga', 'real?'},
'low_fr': 1.0,
'min_units': 10,
...
keep_mask = np.array([q not in quality_exclude and q != '' and q != 'nan'
                      for q in all_qualities])
...
mean_fr = np.mean(trialdat, axis=(1, 2))
fr_mask = mean_fr > params['low_fr']
...
if n_neurons < params['min_units']:
    return None
```

iii. The notes justify the first two filters from the reference code/tutorial and paper, and separately state that sessions should have at least 10 units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For each kept trial, the AI subtracts the trial's go-cue time from every spike time for that trial before binning.

ii.
```python
align_times = ev[params['align_event']]
...
spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
```

iii. The notes explicitly say "Align to goCue" because that is both the decoder instruction and the paper's default alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins over a 5 s window, yielding 500 time bins per trial. No later rebinning is applied; everything is processed directly on that 10 ms grid.

ii.
```python
'dt': 1.0 / 100,  # 10 ms bins
...
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
time_axis = edges[:-1] + params['dt'] / 2
```

iii. The AI's notes repeatedly justify 10 ms by citing `WorkingWithDataObjs.m` and treat that tutorial as the authoritative implementation target.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the chosen go-cue-aligned time grid rather than a separate raw data field. The raw contribution is the go-cue event itself through `align_event='goCue'`; the actual input values are the centers of the analysis bins.

ii.
```python
'align_event': 'goCue',
...
time_axis = edges[:-1] + params['dt'] / 2
time_input = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes describe this as a decoder-specific constructed variable: "Time from goCue ... linear ramp from -2.5 to 2.5s."

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes the bin centers of the common time axis and copies that same 1 x T array into every trial.

ii.
```python
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
time_axis = edges[:-1] + params['dt'] / 2
...
time_input = time_axis.astype(np.float32).reshape(1, -1)
sess_input.append(time_input)
```

iii. The AI did not give a deeper justification beyond saying this decoder input should be a continuous time-varying signal aligned to the chosen neural grid.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is exactly the same `time_axis` used when binning the neural activity, so the two are aligned by construction.

ii.
```python
time_axis = edges[:-1] + params['dt'] / 2
...
trialdat = np.zeros((len(good_indices), n_timebins, n_valid_trials), dtype=np.float32)
...
time_input = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes state that the time axis should be shared across neural and behavioral streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. In the implemented code, lick direction is derived only from the behavioral field `bp['R']` on kept trials. The AI does not use `hit`, `miss`, or `no` to infer actual lick direction.

ii.
```python
valid_trial_indices = valid_trials - 1
lick_direction = bp['R'][valid_trial_indices].copy()
```

iii. This conflicts with the notes, which planned to use `obj.bp.R/L`. The trajectory and final code do not show a later justification for reducing lick direction to a binary copy of `R`.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI simply copies the instructed/right-side flag for each kept trial, then broadcasts it across all time bins of that trial. The output has only two classes, left and right; there is no explicit "none/no lick" class.

ii.
```python
lick_direction = bp['R'][valid_trial_indices].copy()
...
lick_dir = np.full((1, n_timebins), int(sess['lick_direction'][t]), dtype=np.int64)
...
'output_values': [
    ['left', 'right'],
```

iii. The notes originally planned a left/right mapping from `R/L`, but the final code gives no explicit justification for omitting misses and ignore trials as a third lick-direction class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `bp['autowater']` flag.

ii.
```python
behavioral_context = 1.0 - bp['autowater'][valid_trial_indices]
```

iii. The notes explicitly say `autowater=1 -> WC`, `autowater=0 -> DR`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI relabels `autowater` into `WC=0` and `DR=1` by taking `1 - autowater`, then repeats that per-trial value across time bins.

ii.
```python
behavioral_context = 1.0 - bp['autowater'][valid_trial_indices]
...
context = np.full((1, n_timebins), int(sess['behavioral_context'][t]), dtype=np.int64)
...
['WC', 'DR'],
```

iii. This is justified directly in the notes' variable-mapping table.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. In the implemented code, outcome is derived only from the per-trial `bp['hit']` flag. The code does not use `miss` or `no`.

ii.
```python
outcome = bp['hit'][valid_trial_indices].copy()
```

iii. The notes had planned a three-way outcome using hit and miss, but the final code contains only this binary copy and no later justification for the simplification.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI stores `hit` as a binary output and broadcasts it across time bins. With the declared output labels, `0` functions as "incorrect/not hit" and `1` as "correct"; there is no separate ignore class.

ii.
```python
outcome = bp['hit'][valid_trial_indices].copy()
...
outc = np.full((1, n_timebins), int(sess['outcome'][t]), dtype=np.int64)
...
['incorrect', 'correct'],
```

iii. The notes do not justify collapsing miss and ignore together; this appears to be an implementation simplification.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI derives tongue velocity from the bottom-camera tracked feature `top_tongue` in `obj.traj`, using its x/y coordinates and `frameTimes`, plus the session video offset and go-cue times.

ii.
```python
tongue_vel = _compute_feature_velocity_generic(
    traj_data, ntrials, view=2, feat_name='top_tongue',
    vidshift=vidshift, align_times=align_times,
    time_axis=time_axis, is_tongue=True)
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly planned to use the bottom camera for tongue velocity and mentioned "top_tongue or similar."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI linearly interpolates tongue x and y positions from frame times onto the common 10 ms neural grid, computes `np.gradient` on those interpolated positions, sets NaN tongue gradients to zero, and takes Euclidean speed. It does not use the confidence/likelihood channel, does not smooth coordinates, and does not combine the side and bottom tongue views.

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

iii. The notes justify this only at a high level as Euclidean tongue velocity from DLC; the later trajectory shows the AI debugging DLC indexing rather than defending the processing choice itself.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI thresholds tongue speed at the session 50th percentile of all non-NaN values and returns only two categories. NaN values are reassigned to class `0` rather than a separate "not visible" code.

ii.
```python
threshold = np.percentile(valid_vals, 50)
disc = (data >= threshold).astype(np.int64)
disc[np.isnan(data)] = 0
...
'output_values': [
    ...
    ['low', 'high'],
```

iii. The notes justify the 50th-percentile split, but the final code does not justify dropping the requested third "not visible" class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI subtracts a session-wide video offset and the trial's go-cue time from video frame times, then interpolates the trajectory onto the common neural `time_axis`.

ii.
```python
vidshift = vid_file_offset - bit_start
...
ft_aligned = ft - vidshift - align_times[trix]
...
xpos = interp1d(ft_a[valid], x_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
```

iii. The notes explicitly say video should be shifted by the behavior/SpikeGLX offset and aligned to go cue; they also mention a fallback idea of subtracting 0.5 s if needed.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom-camera tracked feature `top_paw` in `obj.traj`, using x/y coordinates, frame times, video offset, and go-cue times.

ii.
```python
paw_vel = _compute_feature_velocity_generic(
    traj_data, ntrials, view=2, feat_name='top_paw',
    vidshift=vidshift, align_times=align_times,
    time_axis=time_axis, is_tongue=False)
```

iii. The notes explicitly planned to use the bottom-camera paw feature.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI interpolates paw x/y positions onto the 10 ms grid, fills missing positions with nearest values, computes `np.gradient`, subtracts median position differences as a baseline, fills missing gradients with nearest values, and takes Euclidean speed. It does not apply likelihood-based masking or Gaussian smoothing.

ii.
```python
xpos = interp1d(...)(time_axis)
ypos = interp1d(...)(time_axis)
...
xpos = _fill_nearest(xpos)
ypos = _fill_nearest(ypos)
...
xvel = np.gradient(xpos)
yvel = np.gradient(ypos)
...
base_xvel = np.nanmedian(np.diff(xpos))
base_yvel = np.nanmedian(np.diff(ypos))
xvel -= (base_xvel if not np.isnan(base_xvel) else 0)
yvel -= (base_yvel if not np.isnan(base_yvel) else 0)
```

iii. The notes justify Euclidean paw velocity from DLC, but the detailed processing choices appear to be implementation simplifications rather than explicitly defended decisions.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI uses the same `_discretize_velocity` function as for tongue velocity: median split across all non-NaN session values, with NaNs mapped to `0`, leaving only two classes.

ii.
```python
paw_vel_disc = _discretize_velocity(paw_vel, n_timebins, n_valid_trials)
...
disc = (data >= threshold).astype(np.int64)
disc[np.isnan(data)] = 0
```

iii. The notes justify the 50th-percentile split but not the missing third "not visible" category.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The AI aligns paw trajectories by subtracting video offset and trial go cue, then interpolating positions onto the same `time_axis` used for neural data.

ii.
```python
ft_aligned = ft - vidshift - align_times[trix]
...
xpos = interp1d(ft_a[valid], x_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
```

iii. This follows the same justification the AI gave for tongue/video alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<anm>_<date>.mat` file. The AI uses per-trial motion-energy traces from that file and side-camera frame times from `traj_data` for alignment.

ii.
```python
me_fn = os.path.join(data_dir, f'motionEnergy_{anm}_{date}.mat')
...
me_file = sio.loadmat(me_fn, squeeze_me=True)
me_struct = me_file['me']
me_cell = me_struct['data'].item()
...
side_cam = traj_data['views'][0] if traj_data is not None else None
```

iii. The notes justify using the standalone motion-energy files and explicitly say to interpolate them to the neural time axis.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI linearly interpolates each per-trial motion-energy trace from frame times onto the 10 ms neural grid, then fills internal NaNs with nearest values. It does not simply bin-average frame values.

ii.
```python
me_aligned[:, trix] = interp1d(ft_aligned[valid], me_trial[valid],
                               kind='linear', bounds_error=False,
                               fill_value=np.nan)(time_axis).astype(np.float32)
...
if not np.isnan(col).all() and np.isnan(col).any():
    me_aligned[:, trix] = _fill_nearest(col)
```

iii. The notes say "Motion energy: Load from motionEnergy files, interpolate to neural time axis," which is the clearest stated justification for this choice.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI applies the same median-split discretizer as for the velocity outputs and again uses only two classes, with NaNs forced to class `0`.

ii.
```python
me_disc = _discretize_velocity(me_data, n_timebins, n_valid_trials)
...
threshold = np.percentile(valid_vals, 50)
disc = (data >= threshold).astype(np.int64)
disc[np.isnan(data)] = 0
```

iii. The notes justify the 50th-percentile threshold but do not justify omitting the requested "no video" class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI subtracts video offset and go-cue time from frame times and interpolates motion energy onto the neural `time_axis`, using side-camera frame times when available and a 400 Hz synthetic clock otherwise.

ii.
```python
if ft is None or ft.size == 0 or np.all(np.isnan(ft)):
    ft = np.arange(1, me_trial.size + 1) / 400.0
...
ft_aligned = ft - vidshift - align_times[trix]
...
interp1d(...)(time_axis)
```

iii. The notes say motion energy should be aligned by video offset and interpolated to the neural time axis.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handles missing data by fallback and imputation rather than by preserving explicit missingness. Missing or all-NaN frame times are replaced by a synthetic `np.arange(...)/400` clock. Missing kinematic positions for paw are nearest-filled before and after differentiation; motion-energy gaps are nearest-filled after interpolation; tongue NaN gradients are zeroed. If a feature cannot be loaded at all, `_discretize_velocity` returns all zeros.

ii.
```python
if ft is None or ft.size == 0 or np.all(np.isnan(ft)):
    ft = np.arange(1, me_trial.size + 1) / 400.0
...
xpos = _fill_nearest(xpos)
ypos = _fill_nearest(ypos)
...
me_aligned[:, trix] = _fill_nearest(col)
...
if data is None:
    return np.zeros((n_timebins, n_trials), dtype=np.int64)
```

iii. The notes mention "fill missing values" for kinematics and document edge cases like missing video, but the trajectory does not show a strong explicit justification beyond trying to keep decoder inputs/outputs finite.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming implemented steps are the nested neural loops over neurons and valid trials in `load_session`, plus the per-trial interpolation and gradient calculations for tongue, paw, and motion energy. File loading is also substantial, but the code's biggest pure-Python cost is the repeated trial-by-trial histogram/interpolation work.

ii.
```python
for neuron_idx, clu_idx in enumerate(good_indices):
    ...
    for t_idx, trial_num in enumerate(valid_trials):
        ...
        counts, _ = np.histogram(spk_times, bins=edges)

for trix in range(ntrials):
    ...
    me_aligned[:, trix] = interp1d(...)(time_axis)

for trix in range(min(ntrials, len(view_data['trials']))):
    ...
    xpos = interp1d(...)(time_axis)
```

iii. The notes only give runtime estimates ("~5s/session") rather than a detailed profiling argument, so this answer is mostly inferred from the implementation structure.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-neuron/per-trial spike binning loop could have been replaced with a session-wide counting approach, and the repeated per-trial interpolation loops for motion energy and kinematics could be partially vectorized or at least batched. The final trial-assembly loop also recreates identical time inputs trial by trial.

ii.
```python
for neuron_idx, clu_idx in enumerate(good_indices):
    for t_idx, trial_num in enumerate(valid_trials):
        ...

for trix in range(ntrials):
    ...

for t in range(n_trials):
    time_input = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The AI did not explicitly justify leaving these loops in place; the code simply reflects a straightforward but non-vectorized implementation.

## 11-c. What processing does the code repeat multiple times?

i. The code repeatedly reconstructs the same per-trial time input array, repeatedly interpolates each trial independently for tongue, paw, and motion energy, and makes separate passes through trajectory data for tongue and paw rather than sharing intermediate aligned positions.

ii.
```python
for t in range(n_trials):
    time_input = time_axis.astype(np.float32).reshape(1, -1)
    sess_input.append(time_input)
...
tongue_vel = _compute_feature_velocity_generic(...)
paw_vel = _compute_feature_velocity_generic(...)
```

iii. There is no explicit justification in the notes; this appears to be a byproduct of keeping the implementation simple.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores `tongue_vel_raw`, `paw_vel_raw`, and `me_raw` in the per-session intermediate results only to support plotting, but those raw arrays are discarded from the final pickle. It also loads some unused raw fields such as `L`, `no`, `sample`, and `delay`, and includes plotting utilities that are not required for downstream decoding.

ii.
```python
return {
    ...
    'tongue_vel_raw': tongue_vel,
    'paw_vel_raw': paw_vel,
    'me_raw': me_data,
}
...
if show_processing and len(all_sessions) <= 2:
    plot_processing(result, f"processing_{result['session_id']}.png")
```

iii. The notes justify the plots as sanity checks, but not as part of the converted dataset itself.
