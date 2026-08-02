# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-coded a session registry, then iterated over that registry and loaded one `data_structure_<animal>_<date>.mat` file plus one matching `motionEnergy_<animal>_<date>.mat` file per session. It combined 25 `Ephys_Behavior` sessions and 19 `RandomizedDelay_Ephys_Behavior` sessions into one dataset.

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

data_path = os.path.join(DATA_ROOT, data_dir, f'data_structure_{anm}_{date}.mat')
obj = mat73.loadmat(data_path)['obj']
...
me_path = os.path.join(DATA_ROOT, data_dir, f'motionEnergy_{anm}_{date}.mat')
```

iii. `CONVERSION_NOTES.md` says the registry was “based on the loading scripts” and explicitly states “Include all 44 ephys sessions”. The trajectory also records the agent’s summary: “All 44 sessions processed successfully.”

## 1-b. How are the data split into subjects?

i. Subjects are split by animal ID (`anm`). The script collects one animal label per session, builds a sorted unique `subjects` list, and creates `subject_idx` by indexing each session’s animal into that list.

ii. ```python
all_animals.append(anm)
subjects_set.add(anm)
...
subjects = sorted(subjects_set)
subject_idx = np.array([subjects.index(anm) for anm in all_animals], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` describes the dataset as 14 unique mice across the selected sessions. No more detailed justification was given beyond mirroring session metadata.

## 1-c. How are the data split into sessions?

i. Each tuple in `ALL_SESSIONS` defines one session by `(animal, date, probes, data_dir)`. `main()` loops over those tuples, calls `process_session(...)`, and appends each successful return as one session entry in `neural`, `input`, and `output`.

ii. ```python
for i, (anm, date, probes, data_dir) in enumerate(sessions):
    result = process_session(anm, date, probes, data_dir, ...)
    if result is None:
        continue
    all_neural.append(result['neural'])
    all_input.append(result['input'])
    all_output.append(result['output'])
```

iii. `CONVERSION_NOTES.md` says the session list was copied from the paper’s loading scripts and that sessions with too few usable trials or neurons are skipped.

## 1-d. How are the data split into trials?

i. Within each session, the code creates a boolean `valid_mask` over all trials, extracts `valid_trials = np.where(valid_mask)[0]`, and then stores one neural matrix, one input array, and one output array per valid trial.

ii. ```python
valid_trials = np.where(valid_mask)[0]
trialdat_valid = trialdat[:, :, valid_trials]
...
for t_idx in range(n_valid):
    neural_trials.append(trialdat_valid[:, :, t_idx].T.astype(np.float32))
    input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
    output_trials.append(out)
```

iii. The notes describe trial curation as excluding early, stimulation, and ignore trials, then saving trial-level matrices after filtering.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if they are not `early`, not `stim.enable`, not `no`, and have either `hit` or `miss` set. The code does not apply the paper’s session-level behavioral inclusion criteria of minimum correct-trial counts per condition.

ii. ```python
valid_mask = ~early & ~stim_enable & ~no
valid_mask = valid_mask & (hit | miss)
valid_trials = np.where(valid_mask)[0]
```

iii. `CONVERSION_NOTES.md` explicitly lists “Trial exclusion: exclude early, stim.enable, and no-response”. The trajectory later claims “Trial filtering ... verified with 0 discrepancies”.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from cluster-level spike metadata: `obj['clu'][probe]['trial']` and `obj['clu'][probe]['trialtm']`, aligned by `obj['bp']['ev']['goCue']`.

ii. ```python
trial_arr = np.array(clu_probe['trial'][clu_idx]).flatten()
trialtm_arr = np.array(clu_probe['trialtm'][clu_idx]).flatten()
goCue = np.array(ev['goCue']).flatten()
aligned = trialtm_arr[spk_mask] - align_times_all[j]
```

iii. The notes map “obj.clu (spike times)” to `neural` and cite `alignSpikes`, `getSeq`, and `removeLowFRClusters` from the reference MATLAB code.

## 2-b. How is the `neural` data processed?

i. For each neuron and trial, spike times are aligned to go cue, histogrammed into 10 ms bins over `[-2.5, 2.5]` s, converted to firing rates, and smoothed with a causal Gaussian window of 15 bins using reflect padding.

ii. ```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
counts, _ = np.histogram(aligned, bins=edges)
fr = counts.astype(np.float32) / DT
trialdat[:, neuron_offset + i, j] = causal_gaussian_smooth(fr, SMOOTH_WINDOW, BC_TYPE)
```

iii. `CONVERSION_NOTES.md` states this was chosen to match `alignSpikes.m`, `getSeq.m`, and `mySmooth.m`, with `dt = 1/100`, `smooth = 15`, and `bctype = 'reflect'`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code excludes cluster qualities `garbage`, `gabrga`, `noisy`, and `real?`, removes neurons with mean firing rate `<= 1 Hz`, and drops whole sessions with fewer than 10 neurons after filtering.

ii. ```python
EXCLUDED_QUALITIES = {'garbage', 'gabrga', 'noisy', 'real?'}
...
if q_stripped.lower() not in {e.lower() for e in excluded_qualities}:
    valid.append(i)
...
keep = mean_fr > low_fr
...
if n_neurons_final < 10:
    return None
```

iii. The notes cite `findClusters.m`, `removeLowFRClusters.m`, and the paper’s “at least 10 units” rule. The chosen FR threshold was justified by the paper text “firing rates exceeding 1 Hz”.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is done per trial by subtracting that trial’s `goCue` timestamp from each spike time before binning.

ii. ```python
ALIGN_EVENT = 'goCue'
...
align_times_all = goCue
...
aligned = trialtm_arr[spk_mask] - align_times_all[j]
```

iii. Both the notes and the trajectory repeatedly state that the decoder should be “aligned to goCue”.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins (`DT = 1/100`) and no additional temporal rebinning beyond the initial histogramming onto that grid.

ii. ```python
DT = 1.0 / 100  # 10 ms bins
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
```

iii. `CONVERSION_NOTES.md` says the agent chose 10 ms because `WorkingWithDataObjs.m` uses `1/100`, while noting that some other scripts use 5 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is derived from the chosen alignment event (`goCue`) plus the fixed window parameters `TMIN`, `TMAX`, and `DT`. The stored input is a common aligned time axis rather than a per-trial transformation of another raw variable.

ii. ```python
goCue = np.array(ev['goCue']).flatten()
...
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. The notes map the decoder input to “time from goCue (s)” and describe it as a continuous aligned time axis.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code constructs bin edges from `-2.5` to `2.5` s and stores the bin centers as the single input channel.

ii. ```python
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. No deeper justification appears in the trajectory beyond keeping input and neural time bases identical.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same `time_axis` used for binned neural activity is written into every trial’s `input`, so alignment is exact by construction.

ii. ```python
time_axis = edges[:-1] + DT / 2
...
neural_trials.append(trialdat_valid[:, :, t_idx].T.astype(np.float32))
input_trials.append(time_axis.reshape(1, -1).astype(np.float32))
```

iii. The notes describe the input as “time from goCue” and the neural data as binned on that same axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived from the behavioral trial labels `R`, `L`, `hit`, and `miss`.

ii. ```python
R = np.array(bp['R']).flatten().astype(bool)
L = np.array(bp['L']).flatten().astype(bool)
hit = np.array(bp['hit']).flatten().astype(bool)
miss = np.array(bp['miss']).flatten().astype(bool)
lick_right = (R & hit) | (L & miss)
```

iii. `CONVERSION_NOTES.md` explicitly documents this mapping.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The agent interpreted “chosen lick side” rather than instructed side: right is coded for `R&hit` or `L&miss`, left for `L&hit` or `R&miss`, then broadcast across time within each trial.

ii. ```python
lick_right = (R & hit) | (L & miss)
lick_direction = lick_right[valid_trials].astype(np.int32)
...
out[0, :] = lick_direction[t_idx]
```

iii. The notes call this out directly as the intended mapping for decoder output.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived from `obj['bp']['autowater']`.

ii. ```python
autowater = np.array(bp['autowater']).flatten().astype(bool)
context = (~autowater[valid_trials]).astype(np.int32)
```

iii. The notes state “autowater=1 -> WC(0); autowater=0 -> DR(1)”.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code inverts the `autowater` flag so that DR becomes `1` and WC becomes `0`, then broadcasts the result across time. This is applied to all retained sessions, including randomized-delay sessions where context is effectively always DR.

ii. ```python
context = (~autowater[valid_trials]).astype(np.int32)
...
out[1, :] = context[t_idx]
```

iii. The notes justify this from the task instructions and also record a separate decision to “include all sessions”, even though the paper’s context analyses are a smaller cohort.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `hit` and, implicitly, the prior filtering to `hit | miss` trials.

ii. ```python
hit = np.array(bp['hit']).flatten().astype(bool)
...
valid_mask = valid_mask & (hit | miss)
outcome = hit[valid_trials].astype(np.int32)
```

iii. `CONVERSION_NOTES.md` documents the mapping `hit -> correct(1); miss -> incorrect(0)`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. After filtering away ignore trials, the code uses `hit` as a binary correctness label and broadcasts it across all time bins in the trial.

ii. ```python
outcome = hit[valid_trials].astype(np.int32)
...
out[2, :] = outcome[t_idx]
```

iii. No further justification was given beyond following the decoder task specification.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The agent derives tongue velocity from the bottom-camera DLC trajectory `obj['traj'][1]['ts']`, using the first tongue-related feature it finds, preferably `top_tongue`, plus `frameTimes` and the video offset.

ii. ```python
traj_bottom = obj['traj'][1]
...
if name == 'top_tongue':
    tongue_idx = i
...
x = ts[:, 0, tongue_idx].copy()
y = ts[:, 1, tongue_idx].copy()
```

iii. The notes say “Use tip-of-tongue displacement from bottom cam view” and the trajectory shows the agent debugging tongue NaNs and deciding not to fill missing tongue values.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code computes the Euclidean speed magnitude from frame-to-frame `x` and `y` gradients at 400 Hz, keeps values only where tongue coordinates are visible, interpolates that speed onto the neural time axis, then converts missing aligned values to zero.

ii. ```python
valid = ~np.isnan(x) & ~np.isnan(y)
vx_all = np.gradient(x) * VIDEO_FR
vy_all = np.gradient(y) * VIDEO_FR
speed = np.sqrt(vx_all**2 + vy_all**2)
speed[~valid] = np.nan
...
f_interp = interp1d(old_time, speed, kind='linear', ...)
tongue_vel[:, trix] = f_interp(time_axis)
tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)
```

iii. The trajectory explicitly says: “The paper says tongue missing values should NOT be filled” and later “tongue not visible = no tongue movement”.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The aligned tongue-speed matrix is thresholded once per session at the 50th percentile of all non-NaN values, except that if the percentile is exactly zero the threshold is replaced with `eps`, forcing exact zeros into the low bin.

ii. ```python
threshold = np.percentile(all_vals, percentile)
if threshold == 0:
    threshold = np.finfo(np.float32).eps
result = (data_2d >= threshold).astype(np.int32)
```

iii. The trajectory shows the explicit justification: median-zero thresholding had made tongue velocity “100% high”, so the agent added the epsilon special case.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue speed is aligned by interpolating bottom-camera frame times minus `vidshift` minus trial-specific `goCue` onto the same `time_axis` used for the neural bins.

ii. ```python
ft = np.array(traj_bottom['frameTimes'][trix]).flatten()
old_time = ft - vidshift - align_times[trix]
...
f_interp = interp1d(old_time, speed, kind='linear', ...)
tongue_vel[:, trix] = f_interp(time_axis)
```

iii. The notes say all behavior streams should be “aligned to neural time axis” via the same video-offset correction used in the MATLAB code.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera DLC tracks in `obj['traj'][1]['ts']` for every feature name containing `paw`, together with `frameTimes` and video offset.

ii. ```python
paw_indices = []
for i, name in enumerate(feat_names):
    if 'paw' in name.lower():
        paw_indices.append(i)
...
x = ts[:, 0, pidx].copy()
y = ts[:, 1, pidx].copy()
```

iii. The notes say “Use paw position from bottom cam, compute velocity, take magnitude.”

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature, NaNs in position are filled with nearest values, `x` and `y` gradients are computed at 400 Hz, the Euclidean speed magnitude is taken, and then speed is averaged across all paw features before interpolation to neural time.

ii. ```python
for arr in [x, y]:
    ...
    arr[nans] = arr[valid[nearest]]
vx = np.gradient(x) * VIDEO_FR
vy = np.gradient(y) * VIDEO_FR
speeds.append(np.sqrt(vx**2 + vy**2))
avg_speed = np.mean(speeds, axis=0)
```

iii. The choice follows the notes’ simplified plan rather than the full reference kinematics pipeline.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity uses the same `discretize_per_session(...)` function as tongue velocity: 50th percentile across the session, with the same zero-threshold-to-`eps` override.

ii. ```python
paw_vel_valid = paw_vel[:, valid_trials]
paw_vel_disc = discretize_per_session(paw_vel_valid)
```

iii. The notes say paw velocity should be “discretize[d] at 50th percentile per session”; the zero-threshold override is justified only in the trajectory’s tongue-debugging discussion, but the helper is reused for paw as well.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw speed is interpolated from bottom-camera frame times, after subtracting both the per-session video offset and each trial’s go cue, onto `time_axis`.

ii. ```python
old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, avg_speed, kind='linear', ...)
paw_vel[:, trix] = f_interp(time_axis)
```

iii. No separate justification was given beyond using the same alignment strategy for all video-derived signals.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from the separate `motionEnergy_<animal>_<date>.mat` file, specifically the `me` variable or nested `me.data.data` structure, plus session video timing from `traj.frameTimes`, `bp.ev.bitStart`, and `sglx.bitcode.bitstart`.

ii. ```python
me_file = scipy.io.loadmat(me_path)
me_var = me_file['me']
...
me_raw = inner['data']
...
bitStart = scipy_stats.mode(np.array(obj['bp']['ev']['bitStart']).flatten(), keepdims=False).mode
bc_bitstart = np.array(obj['sglx']['bitcode']['bitstart']).flatten()
```

iii. The notes say this matches `loadMotionEnergy.m` and `findVideoOffset.m`. The trajectory records debugging of nested and direct-cell-array motion-energy formats.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code linearly interpolates the precomputed motion-energy trace from camera frame times to the neural time base after subtracting `vidshift` and trial `goCue`, then fills NaNs with nearest values.

ii. ```python
old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, me_trial, kind='linear', bounds_error=False, fill_value=np.nan)
me_aligned[:, trix] = f_interp(time_axis)
...
col[nans] = col[valid_idx[nearest]]
```

iii. `CONVERSION_NOTES.md` directly states that motion energy should be “align[ed] to neural time axis, interpolate[d] to dt”.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The aligned motion-energy trace is discretized per session at the 50th percentile via the same generic helper used for tongue and paw outputs, rather than using the manual `me.moveThresh` from the paper.

ii. ```python
me_valid = me_aligned[:, valid_trials]
me_disc = discretize_per_session(me_valid)
```

iii. The notes explicitly say “Motion energy discretization: Per-session 50th percentile threshold,” which follows the decoder task specification rather than the paper’s manual threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned exactly like the other video streams: camera frame time minus video offset minus trial go cue, then interpolated to `time_axis`.

ii. ```python
old_time = ft - vidshift - align_times[trix]
...
me_aligned[:, trix] = f_interp(time_axis)
```

iii. The notes cite the reference `loadMotionEnergy` alignment formula and the trajectory says the final motion-energy output was “bit-for-bit match[ed] with independent recomputation”.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code has several fallback paths: v7.3 `.mat` files are loaded with `mat73`, older MATLAB files with a custom `_load_v5_session`; missing `stim.enable` defaults to all false; missing video `frameTimes` are replaced with `np.arange(...)/400`; motion-energy NaNs and paw NaNs are nearest-filled; tongue NaNs become zeros after interpolation; missing motion-energy files produce an all-zero output; sessions with too few valid trials or fewer than 10 neurons are skipped.

ii. ```python
try:
    obj = mat73.loadmat(data_path)['obj']
except TypeError:
    obj = _load_v5_session(data_path)
...
bp['stim'] = {'enable': np.zeros(int(bp_raw['Ntrials'].flat[0]))}
...
ft = np.arange(1, nframes + 1) / VIDEO_FR
...
tongue_vel = np.nan_to_num(tongue_vel, nan=0.0)
...
else:
    me_disc = np.zeros((n_time, n_valid), dtype=np.int32)
```

iii. The notes justify several of these explicitly, especially MATLAB format handling and the tongue missing-data policy. The trajectory also records a `_load_v5_session()` bug fix.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are session loading and, especially, the nested spike-binning loops over probes, neurons, and trials. Behavioral interpolation over every trial is secondary but still nontrivial.

ii. ```python
for probe_idx, valid_clu in all_cluster_indices:
    for i, clu_idx in enumerate(valid_clu):
        ...
        for j in range(ntrials_total):
            ...
            counts, _ = np.histogram(aligned, bins=edges)
...
print(f"  Spike binning: {t_bin:.1f}s for {total_neurons} neurons")
```

iii. The notes summarize the full conversion as taking about 300 s for 44 sessions, and the script itself prints explicit timing for loading, spike binning, and behavioral processing.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the triple nested loops for spike binning and the per-trial NaN-filling / interpolation loops for motion energy, tongue velocity, and paw velocity.

ii. ```python
for probe_idx, valid_clu in all_cluster_indices:
    for i, clu_idx in enumerate(valid_clu):
        for j in range(ntrials_total):
            ...

for trix in range(ntrials):
    ...
    me_aligned[:, trix] = f_interp(time_axis)
```

iii. The agent did not explicitly discuss vectorization in the notes, so this is an inference from the code structure.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats the same alignment/interpolation pattern for tongue, paw, and motion energy, and it also recomputes bin edges/time axes inside `process_session` even though they are session-invariant.

ii. ```python
old_time = ft - vidshift - align_times[trix]
f_interp = interp1d(old_time, ..., kind='linear', bounds_error=False, fill_value=np.nan)
...
edges = np.arange(TMIN, TMAX + DT, DT)
time_axis = edges[:-1] + DT / 2
```

iii. No explicit justification was given; this is directly visible in the implementation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads `me_thresh` but never uses it, defines `align_and_bin_spikes` and `remove_low_fr_clusters` helpers that are not used in the actual session path, computes rich continuous behavior signals only to discard them after binarization, and broadcasts per-trial categorical outputs across every time bin even though they are constant within trial.

ii. ```python
obj, me_raw, me_thresh, t_load = load_session_data(...)
...
def align_and_bin_spikes(...):
    ...
def remove_low_fr_clusters(...):
    ...
out[0, :] = lick_direction[t_idx]
out[1, :] = context[t_idx]
out[2, :] = outcome[t_idx]
```

iii. The notes do not mention these redundancies; this is inferred from the final code path.
