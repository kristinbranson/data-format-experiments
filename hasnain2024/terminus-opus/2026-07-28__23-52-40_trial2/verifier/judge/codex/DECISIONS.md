# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script hard-codes the 25 `Ephys_Behavior` sessions in `SESSION_META`, opens one HDF5 `.mat` file per session with `h5py`, and processes each listed probe. It does not discover files dynamically; it reconstructs the paper's session list from the MATLAB `loadXXX_ALMVideo.m` scripts.

ii. ```python
DATA_DIR = 'data/Ephys_Behavior'

SESSION_META = [
    ('EKH1', '2021-08-07', [1]),
    ('EKH3', '2021-08-11', [1, 2]),
    ...
    ('JGR3', '2021-11-18', [1]),
]

for idx, (animal, date, probes) in enumerate(sessions):
    result = process_session(animal, date, probes,
                           show_processing=args.show_processing,
                           session_idx=idx)
```

iii. `CONVERSION_NOTES.md` Step 5 says the agent chose `Ephys_Behavior` because it contains the DR+WC two-context task needed for the decoder outputs, and the trajectory shows it copied probe assignments from the MATLAB `load...ALMVideo.m` files.

## 1-b. How are the data split into subjects?

i. Each session carries an `animal` string from `SESSION_META`. After processing, the script builds a sorted unique subject list and stores one subject index per session.

ii. ```python
subjects = sorted(list(set(r['animal'] for r in all_results)))
...
subj_idx = subjects.index(result['animal'])
subject_idx_list.append(subj_idx)
```

iii. The notes repeatedly describe subject identity as the animal name embedded in the session metadata and compare the resulting count against the paper's mouse count.

## 1-c. How are the data split into sessions?

i. Each `(animal, date)` pair is treated as one session. `process_session()` loads one `data_structure_<animal>_<date>.mat` file and one matching motion-energy file.

ii. ```python
session_id = f'{animal}_{date}'
data_file = os.path.join(DATA_DIR, f'data_structure_{session_id}.mat')
me_file = os.path.join(DATA_DIR, f'motionEnergy_{session_id}.mat')
```

iii. The trajectory shows the agent enumerated 25 session files in `data/Ephys_Behavior` and matched them to the MATLAB loaders one-by-one.

## 1-d. How are the data split into trials?

i. Within a session, the script reads `obj.bp.Ntrials` and allocates neural and behavioral arrays across all trial indices. It later packages only `valid_trials`, producing one neural matrix and one input/output pair per kept trial.

ii. ```python
Ntrials = int(bp['Ntrials'][0, 0])
...
trialdat = np.zeros((N_TIMEBINS, len(cluid), Ntrials), dtype=np.float32)
...
for trial_idx in valid_trials:
    neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32))
```

iii. The notes describe the target as session -> trial lists, and the trajectory shows the agent mirrored `obj.trialdat` from the MATLAB code before subsetting valid trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept if they are not early-lick, not ignore/no-response, not stimulation trials, and are either hit or miss trials. Sessions with fewer than 2 valid trials are skipped to satisfy the decoder format requirement.

ii. ```python
valid_mask = (early == 0) & (no == 0) & (stim_enable == 0) & ((hit == 1) | (miss == 1))
valid_trials = np.where(valid_mask)[0]

if len(valid_trials) < 2:
    ...
    return None
```

iii. `CONVERSION_NOTES.md` Step 3 says trial curation excludes early lick, ignore, and stimulation trials. The methods excerpt in the trajectory also says early and ignore trials were omitted from analyses.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from the spike cluster structure under `obj.clu`: cluster qualities plus per-spike `trialtm` and `trial` arrays, aligned using behavioral `goCue` times.

ii. ```python
clu_ref = f['obj']['clu'][probe_idx, 0]
clu_group = f[clu_ref]
...
tm_ref = clu_group['trialtm'][clu_idx, 0]
trial_ref = clu_group['trial'][clu_idx, 0]
spike_trialtm = f[tm_ref][:].flatten()
spike_trial = f[trial_ref][:].flatten().astype(int)
goCue = bp['ev']['goCue'][:].flatten()
```

iii. The notes map spike times in `clu` to `neural` and cite the MATLAB pipeline `alignSpikes -> getSeq -> removeLowFRClusters`.

## 2-b. How is the `neural` data processed?

i. For each kept cluster, spikes are aligned to go cue, histogrammed into 10 ms bins over `[-2.5, 2.5]`, converted to firing rate, and smoothed with a causal Gaussian window of length 15 using reflect padding. Dual-probe sessions are concatenated across neurons.

ii. ```python
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
...
spike_goCue = goCue[spike_trial_valid - 1]
spike_aligned = spike_trialtm_valid - spike_goCue
counts, _ = np.histogram(trial_spikes, bins=EDGES)
fr = counts / PARAMS['dt']
fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
...
trialdat = np.concatenate(all_trialdat, axis=1)
```

iii. `CONVERSION_NOTES.md` Step 1 and the trajectory summarize the reference pipeline exactly this way: align spikes, bin at 10 ms, smooth with `mySmooth`, concatenate EKH3's two ALM probes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script first excludes clusters with qualities `garbage`, `gabrga`, `noisy`, or `real?`, while keeping empty/null labels. It then removes neurons with mean firing rate `<= 1 Hz` and drops sessions with fewer than 10 surviving neurons. The FR filter is applied using a mean over `trialdat`, not the reference MATLAB mean-over-PSTH-by-condition calculation.

ii. ```python
PARAMS = {
    'lowFR': 1.0,
    'quality_exclude': ['garbage', 'gabrga', 'noisy', 'real?'],
}
...
cluid = find_clusters(qualities, PARAMS['quality_exclude'])
...
meanFRs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
fr_mask = meanFRs > PARAMS['lowFR']
trialdat = trialdat[:, fr_mask, :]
...
if n_neurons < 10:
    return None
```

iii. The notes justify `1 Hz` from the paper and explain the null-quality fix from `findClusters.m`. The script comments itself admit the FR calculation is a simplification relative to MATLAB `removeLowFRClusters.m`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike is shifted by the `goCue` time of its own trial, so trial time zero is go-cue onset. The packaged neural matrices use that aligned time base.

ii. ```python
spike_goCue = goCue[spike_trial_valid - 1]
spike_aligned = spike_trialtm_valid - spike_goCue
...
counts, _ = np.histogram(trial_spikes, bins=EDGES)
```

iii. The notes and trajectory both point to `params.alignEvent = 'goCue'` and to `alignSpikes.m`, which computes `trialtm_aligned = trialtm - event`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins (`dt = 1/100`). No extra rebinning is applied after the initial histogramming; the only temporal processing is smoothing.

ii. ```python
PARAMS = {
    'dt': 1/100,
    'smooth': 15,
}
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
```

iii. The notes say the agent chose 10 ms because most reference analysis scripts used `dt = 1/100`, and the paper/methods comparison in Step 3 lists 10 ms bins.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is derived from the chosen alignment event (`goCue`) plus the global time-window parameters `tmin`, `tmax`, and `dt`; it is not read from a dedicated raw time variable per trial.

ii. ```python
PARAMS = {
    'alignEvent': 'goCue',
    'tmin': -2.5,
    'tmax': 2.5,
    'dt': 1/100,
}
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
```

iii. The mapping table in `CONVERSION_NOTES.md` Step 5 explicitly maps the input to the aligned time axis.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script builds the bin-center vector once from the fixed window and stores the same `1 x T` float32 row for every trial.

ii. ```python
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
...
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
session_inputs.append(time_input)
```

iii. The notes treat this as a direct representation of the aligned neural time axis rather than a separately processed behavioral signal.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is exactly the same time axis used to bin the aligned spikes and to interpolate kinematics/motion energy, so input and neural data share time-bin centers.

ii. ```python
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
...
counts, _ = np.histogram(trial_spikes, bins=EDGES)
...
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
```

iii. The trajectory shows the agent validated that the input range matched `[-2.5, 2.5]` and the neural matrices all had 500 aligned bins.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived from the behavioral right-trial indicator `obj.bp.R`; left is inferred as 0 when `R` is 0.

ii. ```python
R = bp['R'][:].flatten()
...
lick_direction = R.copy()  # 1=right, 0=left
```

iii. The notes map `R` to lick direction and the decoder specification itself defines left=0, right=1.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. There is essentially no transformation beyond copying the binary `R` flag, casting to `int`, and repeating it across all time bins for each valid trial.

ii. ```python
trial_info.append({
    'lick_direction': int(lick_direction[trial_idx]),
    ...
})
...
lick_dir = np.full(N_TIMEBINS, info['lick_direction'], dtype=np.int64)
```

iii. The notes describe lick direction as a per-trial label, so the agent turned it into a time-constant trial-long output to fit the decoder format.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`, which marks WC trials.

ii. ```python
autowater = bp['autowater'][:].flatten()
...
context = 1 - autowater
```

iii. The notes say `autowater=1` means WC, so the script inverted it to match the required coding WC=0, DR=1.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The script inverts `autowater` so DR becomes 1 and WC becomes 0, records the value per trial, and repeats it across all time bins in the final output tensor.

ii. ```python
context = 1 - autowater  # DR=1, WC=0
...
context = np.full(N_TIMEBINS, info['context'], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly records this mapping decision.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`, with misses implicitly becoming 0 because only hit/miss trials survive trial filtering.

ii. ```python
hit = bp['hit'][:].flatten()
miss = bp['miss'][:].flatten()
...
outcome = hit.copy()
```

iii. The notes map `hit` to outcome and describe the trial filter as keeping only hit and miss trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The script copies the binary `hit` flag, stores it for each valid trial, and expands it across time bins in the output array.

ii. ```python
trial_info.append({
    ...
    'outcome': int(outcome[trial_idx]),
})
...
outcome = np.full(N_TIMEBINS, info['outcome'], dtype=np.int64)
```

iii. The justification in the notes is simply the decoder requirement incorrect=0, correct=1.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-view DLC trajectories in `obj.traj[0]`, specifically the `tongue` feature's x/y coordinates, plus trial `frameTimes`, `NdroppedFrames`, video-neural offset terms, and `goCue`.

ii. ```python
traj_ref = f['obj']['traj'][0, 0]  # view 1 (side)
...
feat_names.append(''.join(chr(c) for c in chars))
...
if name == 'tongue':
    tongue_idx = i
...
ft_ref = traj_group['frameTimes'][trix, 0]
aligned_times = frameTimes - vidshift - goCue[trix]
```

iii. The notes cite `findPosition.m` and `findVelocity.m` and say the agent chose side-view tongue trajectories to match the reference kinematic pipeline.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Per trial, the tongue x/y trajectories are interpolated onto the neural time axis, differentiated with `np.gradient`, NaN velocities are set to zero, and speed is computed as `sqrt(xvel^2 + yvel^2)`.

ii. ```python
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_times[valid], y[valid], bounds_error=False, fill_value=np.nan)
xpos[:, trix] = fx(taxis)
ypos[:, trix] = fy(taxis)
...
xv = np.gradient(xpos[:, trix])
yv = np.gradient(ypos[:, trix])
xv[np.isnan(xv)] = 0
yv[np.isnan(yv)] = 0
speed = np.sqrt(xvel**2 + yvel**2)
```

iii. The notes and trajectory say this was intended to match `findPosition` plus the tongue-specific branch of `findVelocity` in the reference code.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The script computes a per-session 50th-percentile threshold over all tongue-velocity values, but if that threshold is exactly zero it replaces it with the median of the positive values before binarizing with `>= threshold`.

ii. ```python
threshold = np.percentile(all_values, threshold_percentile)
if threshold == 0:
    pos_values = all_values[all_values > 0]
    if len(pos_values) > 0:
        threshold = np.percentile(pos_values, threshold_percentile)
...
disc = (v >= threshold).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 and Step 10 explicitly justify this as an edge-case fix because tongue velocity was often zero when the tongue was not visible.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue position is interpolated at `frameTimes - vidshift - goCue` onto `TIME_AXIS + advance_movement`, so the resulting velocity lives on the same aligned 10 ms grid as the neural activity.

ii. ```python
taxis = TIME_AXIS + PARAMS['advance_movement']
vidshift = find_video_offset(f)
aligned_times = frameTimes - vidshift - goCue[trix]
...
xpos[:, trix] = fx(taxis)
```

iii. The notes say movement data should not be advanced relative to neural data (`advance_movement = 0.0`) because that matches the reference scripts used for these analyses.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the top-view DLC trajectories in `obj.traj[1]`, selecting all feature names containing `'paw'` (in practice `top_paw` and `bottom_paw`), together with `frameTimes`, dropped-frame metadata, video offset, and `goCue`.

ii. ```python
traj_ref = f['obj']['traj'][1, 0]  # view 2 (top)
...
for i, name in enumerate(feat_names):
    if 'paw' in name.lower():
        paw_indices.append(i)
```

iii. The notes say the agent planned to average `top_paw` and `bottom_paw` from the top view, following the reference kinematic helpers.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature, the script interpolates x/y positions to the neural time axis, fills missing position samples, differentiates the trajectories, subtracts separate x and y baseline derivatives, fills missing velocity values, converts to speed, and averages the resulting speeds across paw features.

ii. ```python
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_times[valid], y[valid], bounds_error=False, fill_value=np.nan)
xp = fx(taxis)
yp = fy(taxis)
...
xv = np.gradient(xpos[:, trix])
yv = np.gradient(ypos[:, trix])
basederiv_x = np.nanmedian(np.diff(xpos[:, trix]))
basederiv_y = np.nanmedian(np.diff(ypos[:, trix]))
...
speed = np.sqrt(xvel**2 + yvel**2)
avg_speed = np.mean(all_speeds, axis=0)
```

iii. The notes justify this as a direct translation of `findPosition`/`findVelocity`, although the actual Python code makes a cleaner x/y baseline subtraction than the MATLAB helper shown in the trajectory.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It uses the shared `discretize_continuous()` helper: one per-session 50th-percentile threshold over all paw-speed values, then binarization with `>= threshold`.

ii. ```python
paw_disc = discretize_continuous(result['paw_vel'])
...
threshold = np.percentile(all_values, threshold_percentile)
disc = (v >= threshold).astype(np.float32)
```

iii. The notes state paw velocity should be discretized exactly per the decoder spec using a session-wise median split.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Like tongue velocity, paw trajectories are sampled at `frameTimes - vidshift - goCue` and interpolated onto the common neural `TIME_AXIS` grid before differentiation.

ii. ```python
taxis = TIME_AXIS + PARAMS['advance_movement']
aligned_times = frameTimes - vidshift - goCue[trix]
xp = fx(taxis)
yp = fy(taxis)
```

iii. The notes say all movement streams were aligned the same way as neural data via the reference `findVideoOffset` and `findPosition` logic.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from the separate `motionEnergy_<session>.mat` file, specifically `me.data`, together with video `frameTimes` from `obj.traj`, the video-neural offset, and `goCue`.

ii. ```python
me_raw = sio.loadmat(me_file)
me_struct = me_raw['me'][0, 0]
me_data = me_struct['data']
...
traj_ref = f['obj']['traj'][0, 0]
ft_ref = traj_group['frameTimes'][trix, 0]
frameTimes = f[ft_ref][:].flatten()
```

iii. The notes and trajectory say the agent mirrored `loadMotionEnergy.m`, including the special case where `me.data` is itself a struct.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The script optionally unwraps nested `me.data.data`, truncates each motion-energy trace to match its frame times, interpolates it onto the neural time axis, and fills remaining NaNs before storing the aligned continuous trace.

ii. ```python
if me_data.dtype.names is not None and 'data' in me_data.dtype.names:
    me_data = me_data[0, 0]['data']
...
min_len = min(len(aligned_times), len(me_trial))
aligned_times = aligned_times[:min_len]
me_trial = me_trial[:min_len]
...
f_interp = interp1d(aligned_times[valid], me_trial[valid],
                   bounds_error=False, fill_value=np.nan)
me_interp = f_interp(taxis)
```

iii. The trajectory documents that the nested-struct fix was added specifically because `loadMotionEnergy.m` in the reference code unwraps `me.data` when needed.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. It uses the same per-session 50th-percentile thresholding helper as paw velocity: compute one session-wide median-like threshold and label each time bin `0/1` by whether it is below or at/above that threshold.

ii. ```python
me_disc = discretize_continuous(result['motion_energy'])
...
threshold = np.percentile(all_values, threshold_percentile)
disc = (v >= threshold).astype(np.float32)
```

iii. This follows the decoder specification recorded in the notes' mapping table.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by subtracting `vidshift` and each trial's `goCue` from video frame times, then interpolating onto the same `TIME_AXIS` used by neural data.

ii. ```python
taxis = TIME_AXIS + PARAMS['advance_movement']
vidshift = find_video_offset(f)
aligned_times = frameTimes - vidshift - goCue[trix]
...
me_interp = f_interp(taxis)
```

iii. The notes explicitly say motion energy was interpolated to the neural time axis using video frame times, matching the reference loader.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles several edge cases permissively: null/empty cluster-quality strings are retained, missing or bad video trials are skipped within the movement extractors, absent movement streams fall back to all-zero traces, missing frame times are replaced by a 400 Hz synthetic grid, some NaNs are filled by interpolation, tongue NaN velocities become zero, and nested motion-energy structs are unwrapped. Entire sessions are skipped only when files are missing, no valid clusters remain, fewer than 2 valid trials remain, or fewer than 10 neurons survive FR filtering.

ii. ```python
except:
    quality_str = ''
...
q_clean = q_clean.replace('\x00', '')
if q_clean in exclude_lower:
    continue
...
except:
    frameTimes = np.arange(1, n_frames + 1) / 400.0
...
xv[np.isnan(xv)] = 0
...
if tongue_vel is not None:
    ...
else:
    tongue_vel_trials.append(np.zeros(N_TIMEBINS, dtype=np.float32))
...
if me_data.dtype.names is not None and 'data' in me_data.dtype.names:
    me_data = me_data[0, 0]['data']
```

iii. `CONVERSION_NOTES.md` Step 10 lists the same edge cases: null quality strings, unusual probe indexing, nested motion-energy structs, and the tongue-threshold special case.

## 11-a. What are the most time-consuming steps of the code?

i. The heaviest work is the nested per-cluster, per-trial spike binning/smoothing inside `process_session()`, followed by per-trial interpolation and differentiation for tongue/paw kinematics and motion energy.

ii. ```python
for ci, clu_idx in enumerate(cluid):
    ...
    for trial_num in range(1, Ntrials + 1):
        ...
        counts, _ = np.histogram(trial_spikes, bins=EDGES)
        fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])

for trix in range(Ntrials):
    ...
    xpos[:, trix] = fx(taxis)
    ypos[:, trix] = fy(taxis)
```

iii. The notes' runtime estimates and the trajectory both focus on per-session processing time; those loops dominate the implementation.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the per-trial histogram loop inside each neuron, the per-column convolution loop in `causal_gaussian_smooth()`, repeated per-trial interpolation loops in tongue/paw/motion-energy extraction, and some repeated Python loops that build subject/feature-name lists.

ii. ```python
for j in range(x_padded.shape[1]):
    result[:, j] = convolve(x_padded[:, j], kern, mode='same')
...
for trial_num in range(1, Ntrials + 1):
    ...
for trix in range(Ntrials):
    ...
```

iii. The original task explicitly asked for vectorization where possible; the agent's script stayed mostly literal to the MATLAB structure instead.

## 11-c. What processing does the code repeat multiple times?

i. It repeatedly recomputes video offsets, rereads feature-name tables, redoes frame-time interpolation separately for tongue and each paw feature, reruns session-wise discretization in both `plot_processing()` and final packaging, and repeatedly expands per-trial scalar outputs across all 500 time bins.

ii. ```python
vidshift = find_video_offset(f)
...
feat_ref = traj_group['featNames'][0, 0]
...
tongue_disc = discretize_continuous(tongue)
paw_disc = discretize_continuous(paw)
me_disc = discretize_continuous(me)
...
lick_dir = np.full(N_TIMEBINS, info['lick_direction'], dtype=np.int64)
```

iii. This follows from the agent's choice to keep separate feature-specific functions and to mirror the MATLAB steps directly.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several values are computed but never used downstream: `spike_tm_abs`, `all_cluid`, and `session_idx`; full all-trial neural arrays are built before invalid trials are discarded; and `show_processing` mode stores extra fields only for plotting. It also replicates trial-level labels across every time bin even though they are constant within a trial.

ii. ```python
tm_abs_ref = clu_group['tm'][clu_idx, 0]
spike_tm_abs = f[tm_abs_ref][:].flatten()
...
all_cluid = []
...
for trial_idx in valid_trials:
    neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32))
...
if show_processing:
    result['trialdat_full'] = trialdat
    result['valid_trials'] = valid_trials
```

iii. These are artifacts of a direct translation from the exploratory/plotting workflow described in the notes rather than strict minimization for the final pickle.
