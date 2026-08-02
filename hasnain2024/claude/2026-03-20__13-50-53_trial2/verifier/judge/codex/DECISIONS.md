# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-coded a session manifest (`SESSION_DEFS`) covering 44 electrophysiology sessions from the two ephys datasets, then iterated that manifest in `convert_all()`. For each session it built paths to `data_structure_<animal>_<date>.mat` and `motionEnergy_<animal>_<date>.mat`, detected MATLAB v7.3/HDF5 versus v5 format, loaded the raw Bpod/spike/DLC fields into a common Python dict, and then processed the session. Behavior-only inhibition datasets were not used.

ii. ```python
SESSION_DEFS = [
    ('EKH1', '2021-08-07', [2], 'ephys'),
    ...
    ('JEB24', '2023-11-03', [1], 'random'),
]

def convert_all(session_defs, outfile, show_processing=False):
    all_sessions = []
    for i, (anm, date, probes, ddir) in enumerate(session_defs):
        result = load_session(anm, date, probes, ddir, PARAMS)
        if result is not None:
            all_sessions.append(result)

def load_session(anm, date, probes, data_dir_key, params):
    data_fn = os.path.join(data_dir, f'data_structure_{anm}_{date}.mat')
    me_fn = os.path.join(data_dir, f'motionEnergy_{anm}_{date}.mat')
    fmt = _detect_file_format(data_fn)
    if fmt == 'hdf5':
        raw = _load_raw_data_hdf5(data_fn, probes)
    else:
        raw = _load_raw_data_v5(data_fn, probes)
```

iii. `CONVERSION_NOTES.md` says the session list came from the reference loading scripts and that the agent intentionally used the ephys datasets only (`Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`). The notes justify excluding other folders because they are behavior-only inhibition experiments, not the electrophysiology data used in the paper’s main analyses.

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse ID (`animal`) from each loaded session. After processing sessions, the agent takes the sorted unique animal names, stores them in `subjects`, and records one `subject_idx` per session.

ii. ```python
all_animals = sorted(set(s['animal'] for s in all_sessions))
animal_to_idx = {a: i for i, a in enumerate(all_animals)}
...
subject_idx.append(animal_to_idx[sess['animal']])
...
'subjects': all_animals,
'subject_idx': np.array(subject_idx, dtype=np.int64),
```

iii. The notes describe the dataset as session-organized and state that subject identity should come from the `anm` field / session filename. No alternative grouping rule was used.

## 1-c. How are the data split into sessions?

i. Each `SESSION_DEFS` entry becomes one output session. If a session has two ALM probes, both probes are loaded together and neurons are concatenated within that session rather than being treated as separate sessions.

ii. ```python
for i, (anm, date, probes, ddir) in enumerate(session_defs):
    result = load_session(anm, date, probes, ddir, PARAMS)

for probe_num in probes:
    probe_idx = probe_num - 1
    ...
    all_spike_times.append(...)
    all_spike_trials.append(...)
    all_qualities.append(...)
```

iii. `CONVERSION_NOTES.md` explicitly says dual-probe sessions are concatenated because both probes record ALM, mirroring `loadSessionData.m`, which concatenates probe outputs for two-probe sessions.

## 1-d. How are the data split into trials?

i. Trials are initially taken from the session’s full Bpod trial count (`bp['Ntrials']`). The agent then creates `valid_trials` from a boolean mask and keeps each surviving trial as one decoder trial. Neural data are stored as one `(neurons, time)` matrix per surviving trial, and outputs/inputs are generated trial-by-trial in the same order.

ii. ```python
bp = raw['bp']
ntrials = raw['ntrials']
...
trial_mask = (bp['early'] == 0) & (bp['stim_enable'] == 0)
valid_trials = np.where(trial_mask)[0] + 1
...
for t_idx, trial_num in enumerate(valid_trials):
    ...
    trialdat[neuron_idx, :, t_idx] = fr.astype(np.float32)
...
for t in range(n_trials):
    sess_neural.append(trialdat[:, :, t].astype(np.float32))
```

iii. The notes say the agent wanted one output trial per valid behavioral trial and intentionally kept hit, miss, and no-response trials so outcome could be decoded.

## 1-e. How are trials filtered based on quality controls?

i. The only explicit trial-level filters are `~early` and `~stim.enable`. The agent did not filter out ignore/no-response trials, and it did not apply the paper’s behavioral inclusion criteria such as minimum counts of correct left/right DR and WC trials.

ii. ```python
trial_mask = (bp['early'] == 0) & (bp['stim_enable'] == 0)
valid_trials = np.where(trial_mask)[0] + 1
if len(valid_trials) < 2:
    ...
```

iii. The notes justify this as a decoder-driven decision: keep all non-early, non-stim trials so `outcome` can include incorrect trials. The notes also mention excluding early/stim because that matches common conditions in the MATLAB code.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from per-unit spike times and trial identities in `obj.clu` for the selected probe(s). In the normalized Python dict these are `all_spike_times`, `all_spike_trials`, and `all_qualities` extracted from each unit’s `tm`, `trial`, and `quality` fields.

ii. ```python
for probe_num in probes:
    ...
    spike_times = h5_read_array(f, tm_refs[i]).flatten()
    spike_trials = h5_read_array(f, trial_refs[i]).flatten().astype(np.int64)
    quality = h5_read_string(f, quality_refs[i]).strip().lower()
    all_spike_times.append(spike_times)
    all_spike_trials.append(spike_trials)
    all_qualities.append(quality)
```

iii. The notes summarize the intended neural pipeline as `loadObjs -> findClusters -> alignSpikes -> getSeq -> removeLowFRClusters`, which all operate on `obj.clu` spike trains.

## 2-b. How is the `neural` data processed?

i. For each surviving neuron and trial, spikes are aligned to go cue, histogrammed into 10 ms bins from -2.5 s to 2.5 s, converted to firing rate by dividing by `dt`, and smoothed with a causal Gaussian window of length 15 using reflected padding.

ii. ```python
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
time_axis = edges[:-1] + params['dt'] / 2
...
spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
counts, _ = np.histogram(spk_times, bins=edges)
fr = causal_gaussian_smooth(counts.astype(np.float64) / params['dt'],
                           params['smooth_window'],
                           params['smooth_bctype'])
trialdat[neuron_idx, :, t_idx] = fr.astype(np.float32)
```

iii. The notes say this was chosen to match `WorkingWithDataObjs.m` and the MATLAB `alignSpikes.m`, `getSeq.m`, and `mySmooth.m` pipeline: go-cue alignment, 10 ms bins, causal Gaussian smoothing, firing rates in spikes/s.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered in three stages: exclude bad cluster quality labels, remove neurons with mean firing rate `<= 1 Hz`, and skip sessions that end up with fewer than 10 neurons.

ii. ```python
quality_exclude = params['quality_exclude']
keep_mask = np.array([q not in quality_exclude and q != '' and q != 'nan'
                      for q in all_qualities])
...
mean_fr = np.mean(trialdat, axis=(1, 2))
fr_mask = mean_fr > params['low_fr']
trialdat = trialdat[fr_mask, :, :]
...
if n_neurons < params['min_units']:
    return None
```

iii. The notes explicitly cite the reference rules: exclude `garbage/noisy/gabrga/real?`, use a 1 Hz firing-rate cutoff, and require at least 10 units per session.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is shifted by subtracting that trial’s `bp.ev.goCue`, so time 0 is go cue onset. The output time axis is the bin-center axis for the -2.5 to 2.5 s window around go cue.

ii. ```python
PARAMS = {
    'align_event': 'goCue',
    'tmin': -2.5,
    'tmax': 2.5,
    'dt': 1.0 / 100,
}
...
align_times = ev[params['align_event']]
spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
```

iii. The notes repeatedly say the decoder task and the paper/tutorial both use go cue alignment, so the agent adopted that as the global alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10 ms bins (`dt = 1/100 s`) over a 5 s window, yielding 500 time bins. Spikes are binned directly at 10 ms; video-based features and motion energy are interpolated onto the same 10 ms axis rather than rebinned from existing 10 ms data.

ii. ```python
PARAMS = {
    'dt': 1.0 / 100,  # 10 ms bins
    'tmin': -2.5,
    'tmax': 2.5,
}
...
'metadata': {
    'time_bin_size': PARAMS['dt'] * 1000,
}
```

iii. The notes justify 10 ms bins by citing `WorkingWithDataObjs.m` instead of `getDefaultParams.m` and note that the decoder input/output streams are all forced onto this shared 10 ms axis.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read as a raw stored variable. It is constructed from the chosen alignment event (`bp.ev.goCue`) plus the fixed analysis window parameters `tmin`, `tmax`, and `dt`.

ii. ```python
align_times = ev[params['align_event']]
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
time_axis = edges[:-1] + params['dt'] / 2
```

iii. The notes describe this input as a synthetic time regressor required by the decoder task rather than a direct raw measurement.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The agent creates a single vector of bin centers from -2.5 to 2.5 seconds and uses that same vector for every trial in every session. No further normalization or warping is applied.

ii. ```python
time_axis = edges[:-1] + params['dt'] / 2
...
time_input = time_axis.astype(np.float32).reshape(1, -1)
sess_input.append(time_input)
```

iii. The notes justify this as the simplest representation of 'time from go cue onset' that is exactly aligned with the neural bins.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses the exact same `time_axis` that was used to bin the neural data, so the input’s sample `k` corresponds to neural bin `k` for every trial.

ii. ```python
# Time axis (should be same for all)
time_axis = all_sessions[0]['time_axis']
...
time_input = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes explicitly state the intent was to share one common time base across neural, kinematic, and motion-energy streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. `lick_direction` is taken from the Bpod trial-side variable `bp['R']` after trial filtering. Right trials become 1 and left trials are implicitly 0.

ii. ```python
lick_direction = bp['R'][valid_trial_indices].copy()
```

iii. The notes map lick direction as left=0, right=1 and treat `bp.R` as the direct binary encoding for that target.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. After selecting valid trials, the per-trial direction value is broadcast across all time bins so that `output` has shape `(n_output, n_timepoints)` for every trial.

ii. ```python
lick_dir = np.full((1, n_timebins), int(sess['lick_direction'][t]), dtype=np.int64)
```

iii. The notes say lick direction is a per-trial categorical label, so the agent repeated it across time only to satisfy the decoder format.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp['autowater']`. The agent inverted it so `autowater=1` becomes WC=0 and `autowater=0` becomes DR=1.

ii. ```python
behavioral_context = 1.0 - bp['autowater'][valid_trial_indices]
```

iii. The notes explicitly record this mapping: WC = 0, DR = 1, with `autowater=1` interpreted as the water-cued context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The autowater flag is inverted, restricted to valid trials, cast to integer later, and broadcast across time within each trial.

ii. ```python
context = np.full((1, n_timebins), int(sess['behavioral_context'][t]), dtype=np.int64)
```

iii. The notes justify broadcasting exactly as for lick direction: context is session/trial-level, not time-varying, but the decoder format expects per-trial arrays.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived only from `bp['hit']`. The script does not explicitly combine `hit`, `miss`, and `no`; instead, any trial with `hit=0` is implicitly treated as incorrect.

ii. ```python
outcome = bp['hit'][valid_trial_indices].copy()
```

iii. The notes say the agent wanted `correct=1` and `incorrect=0`, and chose `hit` as the direct source. The notes also state that all non-hit trials were kept so the decoder could learn outcome.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The outcome value is copied from `hit`, then broadcast across all time bins. Misses and no-response/ignore trials are both collapsed into the same `0` category because the code never excludes `bp['no']` or assigns it separately.

ii. ```python
outc = np.full((1, n_timebins), int(sess['outcome'][t]), dtype=np.int64)
```

iii. The notes justify this as a binary decoder target, but they do not mention the paper’s convention of omitting ignore trials from behavioral analyses.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from bottom-camera DLC trajectories for the single feature name `top_tongue` plus per-trial `frameTimes`, the video shift offset, and go-cue times.

ii. ```python
tongue_vel = _compute_feature_velocity_generic(
    traj_data, ntrials, view=2, feat_name='top_tongue',
    vidshift=vidshift, align_times=align_times,
    time_axis=time_axis, is_tongue=True)
```

iii. The notes say the agent deliberately used a bottom-camera tongue feature and describes this as 'from bottom cam'. No stronger justification appears beyond wanting a visible tongue trajectory and obtaining sensible summary statistics.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the agent interpolates `x` and `y` tongue position onto the neural time axis using `frameTimes - vidshift - goCue`, computes first differences with `np.gradient`, sets NaN tongue velocities to 0, and converts to scalar speed with `sqrt(xvel^2 + yvel^2)`.

ii. ```python
xpos = interp1d(ft_a[valid], x_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
ypos = interp1d(ft_a[valid], y_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
...
xvel = np.gradient(xpos)
yvel = np.gradient(ypos)
if is_tongue:
    xvel[np.isnan(xvel)] = 0
    yvel[np.isnan(yvel)] = 0
speed[:, trix] = np.sqrt(xvel**2 + yvel**2).astype(np.float32)
```

iii. The notes cite the paper/methods rule that tongue missing values should not be nearest-filled and that tongue invisibility should map to zero velocity.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The continuous tongue speed values are thresholded at the 50th percentile of all non-NaN tongue-speed samples within the session. Values below threshold become 0 and values at or above threshold become 1.

ii. ```python
def _discretize_velocity(data, n_timebins, n_trials):
    valid_vals = data[~np.isnan(data)]
    threshold = np.percentile(valid_vals, 50)
    disc = (data >= threshold).astype(np.int64)
    disc[np.isnan(data)] = 0
    return disc
```

iii. The notes explicitly say continuous outputs were discretized per session at the 50th percentile because that was required by the decoder task.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. It is aligned by subtracting the estimated video offset and the trial’s go-cue time from the tongue `frameTimes`, then interpolating onto the same 10 ms `time_axis` as the neural data.

ii. ```python
ft_aligned = ft - vidshift - align_times[trix]
...
xpos = interp1d(...)(time_axis)
ypos = interp1d(...)(time_axis)
```

iii. The notes say all behavioral time series should share the neural go-cue-aligned axis, and they mention the 0.5 s/video-bitcode offset logic from the reference code.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera DLC trajectories for the single feature name `top_paw`, together with frame times, video offset, and go-cue alignment.

ii. ```python
paw_vel = _compute_feature_velocity_generic(
    traj_data, ntrials, view=2, feat_name='top_paw',
    vidshift=vidshift, align_times=align_times,
    time_axis=time_axis, is_tongue=False)
```

iii. The notes say paw velocity comes from the bottom camera. They do not explain why `top_paw` was preferred over `bottom_paw` or a combined paw representation.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The agent interpolates paw `x,y` coordinates onto the neural axis, fills missing positions with nearest values, computes gradients, subtracts a per-trial median baseline derivative, fills remaining NaNs in velocity, and converts to Euclidean speed.

ii. ```python
if not is_tongue:
    xpos = _fill_nearest(xpos)
    ypos = _fill_nearest(ypos)
...
base_xvel = np.nanmedian(np.diff(xpos))
base_yvel = np.nanmedian(np.diff(ypos))
xvel -= (base_xvel if not np.isnan(base_xvel) else 0)
yvel -= (base_yvel if not np.isnan(base_yvel) else 0)
xvel = _fill_nearest(xvel)
yvel = _fill_nearest(yvel)
speed[:, trix] = np.sqrt(xvel**2 + yvel**2).astype(np.float32)
```

iii. The notes justify this by citing the kinematics code path: non-tongue features have nearest-filled missing values and baseline-subtracted first derivatives.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw speed is discretized with the same per-session 50th-percentile rule used for tongue and motion energy.

ii. ```python
paw_vel_disc = _discretize_velocity(paw_vel, n_timebins, n_valid_trials)
```

iii. The notes explicitly state all continuous decoder outputs were thresholded per session at the median.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw trajectories are aligned with the neural data by shifting trial frame times by `vidshift` and go-cue time, then interpolating onto the neural `time_axis`.

ii. ```python
ft_aligned = ft - vidshift - align_times[trix]
...
interp1d(ft_a[valid], x_r[valid], ...)(time_axis)
```

iii. The notes say video-derived variables were all synchronized using the same offset-and-interpolation procedure as motion energy.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<animal>_<date>.mat` file, specifically `me.data`, combined with side-camera frame times from `traj_data['views'][0]`, the video offset, and go-cue times.

ii. ```python
me_file = sio.loadmat(me_fn, squeeze_me=True)
me_struct = me_file['me']
me_cell = me_struct['data'].item()
...
side_cam = traj_data['views'][0]
ft = side_cam['trials'][trix]['frameTimes']
```

iii. The notes describe motion energy as coming from the precomputed motion-energy files used by `loadMotionEnergy.m`, not from recomputing framewise image differences in Python.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The agent loads the per-trial motion-energy vectors, aligns them to go cue using video frame times and `vidshift`, interpolates them onto the neural axis, and nearest-fills NaNs within each trial. If loading fails, the session later gets an all-zero discretized motion-energy output because `_discretize_velocity(None, ...)` returns zeros.

ii. ```python
ft_aligned = ft - vidshift - align_times[trix]
...
me_aligned[:, trix] = interp1d(ft_aligned[valid], me_trial[valid],
                               kind='linear', bounds_error=False,
                               fill_value=np.nan)(time_axis).astype(np.float32)
...
if not np.isnan(col).all() and np.isnan(col).any():
    me_aligned[:, trix] = _fill_nearest(col)
```

iii. The notes say this was intended to follow `loadMotionEnergy.m`, including interpolation to the neural axis and nearest filling of edge NaNs.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The final decoder label is not thresholded with the paper’s manual `moveThresh`. Instead, it is discretized at the session-wide 50th percentile of aligned motion-energy values, because the decoder task explicitly requested median-based binarization.

ii. ```python
me_disc = _discretize_velocity(me_data, n_timebins, n_valid_trials)
```

iii. The notes explicitly acknowledge the paper’s manual threshold but say the decoder specification overrides that and requires a 50th-percentile threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses side-camera frame times, subtracts `vidshift` and per-trial go-cue times, and is interpolated onto the exact neural `time_axis`.

ii. ```python
ft_aligned = ft - vidshift - align_times[trix]
...
interp1d(ft_aligned[valid], me_trial[valid], ...)(time_axis)
```

iii. The notes cite `loadMotionEnergy.m` as the template for this alignment.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses many fallbacks: it guesses file format, defaults missing `stim_enable` to zeros, estimates `vidshift` with a fallback of 0.5 s, fabricates frame times when absent, fills many NaNs with nearest values, turns missing tongue velocity samples into zero, skips sessions that fail major loading/QC steps, and converts completely missing continuous outputs to all-zero discrete labels. It does not remove all-zero neural trials that remain after processing.

ii. ```python
vidshift = 0.5
try:
    ...
except Exception:
    pass
...
if ft is None or ft.size == 0 or np.all(np.isnan(ft)):
    ft = np.arange(1, me_trial.size + 1) / 400.0
...
if data is None:
    return np.zeros((n_timebins, n_trials), dtype=np.int64)
```

iii. The notes frame these as pragmatic robustness measures and mention known edge cases such as degenerate motion-energy sessions and zero-neural trials.

## 11-a. What are the most time-consuming steps of the code?

i. The main runtime cost is the nested neuron-by-trial spike-binning and smoothing loop. A secondary cost is per-trial interpolation of motion energy and DLC features. The notes estimate about 5 s per session and the full run log confirms most time is spent in per-session processing, not pickle writing.

ii. ```python
for neuron_idx, clu_idx in enumerate(good_indices):
    ...
    for t_idx, trial_num in enumerate(valid_trials):
        ...
        counts, _ = np.histogram(spk_times, bins=edges)
        fr = causal_gaussian_smooth(...)

for trix in range(ntrials):
    ...
    me_aligned[:, trix] = interp1d(...)(time_axis)

for trix in range(min(ntrials, len(view_data['trials']))):
    ...
    xpos = interp1d(...)(time_axis)
```

iii. `CONVERSION_NOTES.md` explicitly calls out full loading as the dominant runtime and gives a per-session runtime estimate.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The neuron/trial spike loop could be vectorized or rewritten with sparse/event-based batching; the per-trial interpolation loops for motion energy and kinematics could also be batched less expensively; and `causal_gaussian_smooth` still loops over columns during convolution.

ii. ```python
for neuron_idx, clu_idx in enumerate(good_indices):
    for t_idx, trial_num in enumerate(valid_trials):
        ...

for j in range(x_filt.shape[1]):
    out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')

for trix in range(ntrials):
    ...
for trix in range(min(ntrials, len(view_data['trials']))):
    ...
```

iii. The notes do not claim any optimization here; the trajectory shows the agent focused on correctness and decoder validation rather than performance engineering.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats the same alignment/interpolation pattern for motion energy, tongue, and paw signals, and repeats per-trial broadcasting for every static output variable. It also recomputes the same `time_input` for every trial instead of reusing one shared array.

ii. ```python
ft_aligned = ft - vidshift - align_times[trix]
...
interp1d(...)(time_axis)

lick_dir = np.full((1, n_timebins), int(sess['lick_direction'][t]), dtype=np.int64)
context = np.full((1, n_timebins), int(sess['behavioral_context'][t]), dtype=np.int64)
outc = np.full((1, n_timebins), int(sess['outcome'][t]), dtype=np.int64)
time_input = time_axis.astype(np.float32).reshape(1, -1)
```

iii. The notes describe these repeated operations functionally, but do not try to factor them into shared helpers beyond `_discretize_velocity` and `_compute_feature_velocity_generic`.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes and retains raw continuous tongue/paw/motion-energy arrays inside each session result even though the final pickle stores only discretized outputs. It also includes optional plotting/reporting code that is not part of the final dataset and performs summary/statistics work only for validation. More importantly, it processes miss/no-response trials and several sessions with degenerate or missing motion-energy outputs that are not consistent with the paper’s main downstream analyses.

ii. ```python
return {
    ...
    'tongue_vel_disc': tongue_vel_disc,
    'paw_vel_disc': paw_vel_disc,
    'me_disc': me_disc,
    'tongue_vel_raw': tongue_vel,
    'paw_vel_raw': paw_vel,
    'me_raw': me_data,
}

if show_processing and len(all_sessions) <= 2:
    plot_processing(result, f"processing_{result['session_id']}.png")
```

iii. The notes justify the raw arrays and plots as sanity checks, not as required outputs.
