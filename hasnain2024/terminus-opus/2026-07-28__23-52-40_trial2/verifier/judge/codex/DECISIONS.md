# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a 25-session `SESSION_META` list and loads only files from `data/Ephys_Behavior`. For each session it opens `data_structure_<session>.mat` with `h5py.File(...)` and loads motion energy from `motionEnergy_<session>.mat` in the same folder. It does not scan both task folders and does not support the non-HDF5 MATLAB loader used in the human reference.

ii. ```python
DATA_DIR = 'data/Ephys_Behavior'

SESSION_META = [
    ('EKH1', '2021-08-07', [1]),
    ...
    ('JGR3', '2021-11-18', [1]),
]
```

```python
session_id = f'{animal}_{date}'
data_file = os.path.join(DATA_DIR, f'data_structure_{session_id}.mat')
me_file = os.path.join(DATA_DIR, f'motionEnergy_{session_id}.mat')
...
f = h5py.File(data_file, 'r')
```

iii. The notes explicitly justify this as “Use Ephys_Behavior only” because it “Has DR+WC two-context task,” and the trajectory shows the agent deciding to exclude `RandomizedDelay_Ephys_Behavior` and treat the 25 fixed-delay sessions as the target dataset.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `animal` element of each `(animal, date, probes)` tuple in `SESSION_META`. After processing all sessions, the AI builds a sorted unique subject list and a per-session `subject_idx`.

ii. ```python
result = {
    ...
    'session_id': session_id,
    'animal': animal,
}
```

```python
subjects = sorted(list(set(r['animal'] for r in all_results)))
...
subj_idx = subjects.index(result['animal'])
subject_idx_list.append(subj_idx)
```

iii. The notes treat the tuple metadata as authoritative session metadata and report 10 animals from the fixed-delay folder. No deeper justification is given beyond matching the hand-transcribed session table.

## 1-c. How are the data split into sessions?

i. One session is one `(animal, date, probes)` entry in `SESSION_META`, corresponding to one `data_structure_*.mat` file and one `motionEnergy_*.mat` file under `data/Ephys_Behavior`. Each successful session becomes one element in `neural`, `input`, and `output`.

ii. ```python
for idx, (animal, date, probes) in enumerate(sessions):
    result = process_session(animal, date, probes,
                           show_processing=args.show_processing,
                           session_idx=idx)
    if result is not None:
        all_results.append(result)
```

```python
data = {
    'neural': neural_list,
    'input': input_list,
    'output': output_list,
    ...
}
```

iii. The notes justify the session list as coming from the animal-specific `loadXXX_ALMVideo.m` scripts, but they intentionally restrict to the fixed-delay sessions only.

## 1-d. How are the data split into trials?

i. Trials are indexed by the behavioral `bp['Ntrials']` count and by the flattened per-trial behavioral arrays (`L`, `R`, `hit`, `miss`, `no`, `autowater`, `early`, `stim.enable`, `ev.goCue`). Valid trial indices are selected with `np.where(valid_mask)[0]`, and spikes and video streams are later indexed by those trial numbers.

ii. ```python
bp = f['obj']['bp']
Ntrials = int(bp['Ntrials'][0, 0])
...
valid_trials = np.where(valid_mask)[0]  # 0-indexed
```

```python
for trial_idx in valid_trials:
    neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32))
    ...
    tongue_vel_trials.append(tongue_vel[:, trial_idx].astype(np.float32))
```

iii. The code comments treat Bpod trial arrays as the source of truth. The notes summarize this as standard per-trial behavioral indexing rather than reconstructed trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials with `early == 0`, `no == 0`, `stim_enable == 0`, and either `hit == 1` or `miss == 1`. This removes early licks, ignore/no-response trials, and stimulation trials. It does not implement the reference’s extra cutoff for trials that extend beyond the end of the recording.

ii. ```python
valid_mask = (early == 0) & (no == 0) & (stim_enable == 0) & ((hit == 1) | (miss == 1))
valid_trials = np.where(valid_mask)[0]
```

iii. The notes explicitly justify this as “Exclude early, no, stim trials” and “Include hit and miss trials,” claiming this matches the paper’s methods.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from cluster spike times and spike-trial assignments in `obj.clu`: specifically `trialtm`, `trial`, and cluster `quality`. Alignment also uses `bp.ev.goCue`.

ii. ```python
qualities = get_cluster_qualities(f, clu_group)
...
tm_ref = clu_group['trialtm'][clu_idx, 0]
trial_ref = clu_group['trial'][clu_idx, 0]
...
spike_trialtm = f[tm_ref][:].flatten()
spike_trial = f[trial_ref][:].flatten().astype(int)
...
spike_goCue = goCue[spike_trial_valid - 1]
spike_aligned = spike_trialtm_valid - spike_goCue
```

iii. The notes frame this as matching `findClusters.m`, `alignSpikes.m`, and `getSeq.m`: quality-filter clusters, align `trialtm` to `goCue`, then bin and smooth.

## 2-b. How is the `neural` data processed?

i. For each retained cluster, the AI bins aligned spikes into 10 ms bins from -2.5 s to 2.5 s, converts counts to firing rates in spikes/s, applies a causal Gaussian smoother with a 15-sample window, concatenates probes, and stores the per-trial result as `(n_neurons, n_timebins)` arrays.

ii. ```python
PARAMS = {
    'tmin': -2.5,
    'tmax': 2.5,
    'dt': 1/100,
    'smooth': 15,
    'bctype': 'reflect',
}
...
counts, _ = np.histogram(trial_spikes, bins=EDGES)
fr = counts / PARAMS['dt']
fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
```

```python
trialdat = np.concatenate(all_trialdat, axis=1)
...
neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32))
```

iii. The notes repeatedly justify the 10 ms bins and 15-sample causal Gaussian as “matching reference code and paper,” and the trajectory shows the agent explicitly choosing `dt = 1/100`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first excludes clusters whose quality label, after lower-casing and stripping nulls, is one of `garbage`, `gabrga`, `noisy`, or `real?`. It then removes neurons whose mean firing rate over all time bins and trials is `<= 1 Hz`. Finally, it skips an entire session if fewer than 10 neurons remain.

ii. ```python
'quality_exclude': ['garbage', 'gabrga', 'noisy', 'real?'],
'lowFR': 1.0,
```

```python
cluid = find_clusters(qualities, PARAMS['quality_exclude'])
...
meanFRs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
fr_mask = meanFRs > PARAMS['lowFR']
...
if n_neurons < 10:
    print(f'    Skipping {session_id}: only {n_neurons} neurons (need >= 10)')
```

iii. The notes justify the quality filter as matching `findClusters.m`, the firing-rate filter as matching the paper’s 1 Hz threshold, and the `>= 10` neuron rule by quoting a paper statistic rather than a loader rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns spikes to the go cue by subtracting the `goCue` time of the spike’s own trial from each `trialtm` value before binning.

ii. ```python
spike_goCue = goCue[spike_trial_valid - 1]
spike_aligned = spike_trialtm_valid - spike_goCue
```

iii. The code comments and notes explicitly cite the reference formula `trialtm_aligned = trialtm - event` with `event = goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 10 ms bins over a 5 s window (`-2.5` to `2.5`). Neural spikes are directly binned on this grid; the video-derived outputs are interpolated onto the same 10 ms grid rather than rebinned from frame-level data.

ii. ```python
'dt': 1/100,         # 10ms time bins
...
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
N_TIMEBINS = len(TIME_AXIS)
```

iii. The notes explicitly defend `dt = 10ms` as “Standard across analysis scripts,” even though the task asked to match the same processing as the reference solution.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The actual stored input is not computed from a per-trial raw array. It is a fixed `TIME_AXIS` built from `tmin`, `tmax`, and `dt`; conceptually it represents time relative to the go cue because every other stream is aligned to `goCue`.

ii. ```python
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
```

```python
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
session_inputs.append(time_input)
```

iii. The notes describe this as “Time axis” mapped to `input[0]`, with the rationale that the decoder input should be time from go cue onset.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI simply builds the bin centers of the common analysis grid and repeats that same 1-by-time vector for every trial in every session.

ii. ```python
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
...
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
session_inputs.append(time_input)
```

iii. The notes do not give a deeper justification beyond choosing the common neural/video time axis and the 10 ms bin width.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is exactly the same time grid used for neural bin edges and video interpolation, so the AI treats it as the neural time axis itself.

ii. ```python
counts, _ = np.histogram(trial_spikes, bins=EDGES)
...
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
```

iii. The notes state that neural data and movement variables were all aligned to go cue and that the time input uses that same axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. In the final code, lick direction is derived only from `bp['R']` after the trial filter has already removed ignore/no-response trials. The loaded `L`, `hit`, and `miss` arrays are not used to compute the output value itself.

ii. ```python
L = bp['L'][:].flatten()
R = bp['R'][:].flatten()
hit = bp['hit'][:].flatten()
miss = bp['miss'][:].flatten()
...
lick_direction = R.copy()  # 1=right, 0=left
```

iii. The notes justify this mapping as “R (right trial) -> left=0, right=1,” which only works because the code excluded `no` trials earlier.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. No additional computation is applied beyond copying the per-trial right-trial flag and carrying it through as a repeated time series. The output has only two classes, left and right.

ii. ```python
lick_direction = R.copy()  # 1=right, 0=left
...
lick_dir = np.full(N_TIMEBINS, info['lick_direction'], dtype=np.int64)
```

iii. The notes claim this matches the decoder target specification and the paper, but they also explicitly chose to drop ignore trials instead of representing them.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `bp['autowater']` flag.

ii. ```python
autowater = bp['autowater'][:].flatten()  # WC trials
...
context = 1 - autowater  # DR=1, WC=0
```

iii. The notes map `autowater` directly to behavioral context and describe water-cued trials as those with autowater enabled.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI relabels `autowater == 1` as WC (`0`) and everything else as DR (`1`), then repeats that value across time bins for each trial.

ii. ```python
context = 1 - autowater  # DR=1, WC=0 (autowater=1 means WC)
...
context = np.full(N_TIMEBINS, info['context'], dtype=np.int64)
```

iii. The notes explicitly state this coding choice as “WC=0, DR=1.”

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. In the final implementation, outcome is derived only from `bp['hit']` after filtering out ignore/no-response trials. `miss` is used in the validity mask, but not in the final outcome assignment.

ii. ```python
hit = bp['hit'][:].flatten()
miss = bp['miss'][:].flatten()
...
valid_mask = ... & ((hit == 1) | (miss == 1))
...
outcome = hit.copy()  # 1=correct, 0=incorrect
```

iii. The notes justify this as “hit -> correct, miss -> incorrect” and treat the binary coding as acceptable because ignored trials were removed beforehand.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is stored as a binary variable: `1` for hit/correct, `0` otherwise among the retained trials. The value is then repeated across time bins.

ii. ```python
outcome = hit.copy()  # 1=correct, 0=incorrect
...
outcome = np.full(N_TIMEBINS, info['outcome'], dtype=np.int64)
```

iii. The notes say this matches the prompt’s incorrect/correct coding and justify dropping ignored trials during trial curation.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-view camera only: `obj.traj[0]`, its `featNames`, per-trial `ts`, `frameTimes`, and `NdroppedFrames`, plus `bp.ev.goCue` and the session-wide video offset from `bp.ev.bitStart` and `sglx.bitcode.bitstart`.

ii. ```python
traj_ref = f['obj']['traj'][0, 0]  # view 1 (side)
...
if name == 'tongue':
    tongue_idx = i
...
aligned_times = frameTimes - vidshift - goCue[trix]
```

iii. The notes justify this as “Side view (view 1), feature 'tongue'” and describe it as matching `findPosition + findVelocity`.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the AI interpolates tongue x/y positions from frame times onto the neural `TIME_AXIS`, takes discrete gradients of the interpolated positions, replaces NaN gradients with zero, and computes speed as `sqrt(xvel**2 + yvel**2)`. It does not threshold by DLC likelihood, does not smooth x/y before differentiating, and does not combine the second tongue view.

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

iii. The notes claim this matches the reference velocity pipeline and explicitly mention “speed = sqrt(xvel^2 + yvel^2)” for the side-view tongue feature.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI pools all tongue-speed values in a session, takes the 50th percentile as the threshold, and labels each bin as `0` below threshold or `1` at/above threshold. If the median is exactly zero, it recomputes the threshold from only positive values. Missing visibility is not given a separate class.

ii. ```python
threshold = np.percentile(all_values, threshold_percentile)
...
if threshold == 0:
    pos_values = all_values[all_values > 0]
    if len(pos_values) > 0:
        threshold = np.percentile(pos_values, threshold_percentile)
...
disc = (v >= threshold).astype(np.float32)
```

iii. The notes explicitly justify the zero-threshold fallback as an “edge case” for tongue visibility and say this is needed because tongue velocity is mostly zero when the tongue is not visible.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI computes a session-wide video offset from bitcode timing, subtracts both that offset and each trial’s `goCue` from frame times, and then interpolates the tongue coordinates directly onto the neural `TIME_AXIS`.

ii. ```python
vidshift = find_video_offset(f)
...
aligned_times = frameTimes - vidshift - goCue[trix]
...
xpos[:, trix] = fx(taxis)
ypos[:, trix] = fy(taxis)
```

iii. The notes justify this as matching `findVideoOffset.m` and aligning movement “to neural time axis.”

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the top-view camera `obj.traj[1]`, its `featNames`, per-trial `ts`, `frameTimes`, and `NdroppedFrames`, plus `goCue` and the bitcode-derived video offset. The implementation uses every feature whose name contains `"paw"` and averages them.

ii. ```python
traj_ref = f['obj']['traj'][1, 0]  # view 2 (top)
...
for i, name in enumerate(feat_names):
    if 'paw' in name.lower():
        paw_indices.append(i)
```

iii. The notes justify this as using the top view and “top_paw + bottom_paw averaged.”

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature and trial, the AI interpolates x/y coordinates onto the neural time axis, nearest-fills missing coordinates, differentiates the interpolated position, subtracts a baseline derivative, nearest-fills missing derivatives, converts to speed, and averages the resulting paw speeds across all paw features.

ii. ```python
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_times[valid], y[valid], bounds_error=False, fill_value=np.nan)
xp = fx(taxis)
yp = fy(taxis)
...
xp = np.interp(indices, indices[mask], xp[mask])
yp = np.interp(indices, indices[mask], yp[mask])
...
xv = np.gradient(xpos[:, trix])
yv = np.gradient(ypos[:, trix])
basederiv_x = np.nanmedian(np.diff(xpos[:, trix]))
basederiv_y = np.nanmedian(np.diff(ypos[:, trix]))
...
avg_speed = np.mean(all_speeds, axis=0)
```

iii. The notes claim this matches `findVelocity.m` for non-tongue features and explicitly state the decision to average top and bottom paw features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is discretized with the same `discretize_continuous(...)` helper used for tongue and motion energy: per-session median split into binary low/high classes, with the same zero-threshold fallback and no explicit missing-data class.

ii. ```python
paw_disc = discretize_continuous(result['paw_vel'])
...
disc = (v >= threshold).astype(np.float32)
```

iii. The notes justify this as using the required per-session 50th percentile threshold.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The AI aligns paw data exactly like tongue data: frame times are corrected by the session video offset and trial go cue, then positions are interpolated to `TIME_AXIS`.

ii. ```python
aligned_times = frameTimes - vidshift - goCue[trix]
...
xp = fx(taxis)
yp = fy(taxis)
```

iii. The notes state that kinematics were “interpolated to neural time axis,” using the same go-cue-centered timing as neural data.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from the separate `motionEnergy_<session>.mat` file, specifically its `me['data']` field after optional unwrapping of an extra nested struct. Alignment additionally uses side-camera `frameTimes`, `goCue`, and the session video offset.

ii. ```python
me_raw = sio.loadmat(me_file)
me_struct = me_raw['me'][0, 0]
me_data = me_struct['data']
if me_data.dtype.names is not None and 'data' in me_data.dtype.names:
    me_data = me_data[0, 0]['data']
```

```python
traj_ref = f['obj']['traj'][0, 0]
...
aligned_times = frameTimes - vidshift - goCue[trix]
```

iii. The notes explicitly justify the nested-struct unwrapping by citing the MATLAB reference guard `if isstruct(me.data), me.data = me.data.data`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI loads each trial’s frame-level motion energy trace, aligns frame times to go cue, truncates the trace to the shared length of `frameTimes` and `me_trial`, linearly interpolates it onto the neural `TIME_AXIS`, nearest-fills NaNs, and later discretizes the aligned trace.

ii. ```python
me_trial = me_data[trix, 0].flatten()
...
min_len = min(len(aligned_times), len(me_trial))
aligned_times = aligned_times[:min_len]
me_trial = me_trial[:min_len]
...
f_interp = interp1d(aligned_times[valid], me_trial[valid],
                   bounds_error=False, fill_value=np.nan)
me_interp = f_interp(taxis)
...
me_interp = np.interp(indices, indices[mask], me_interp[mask])
```

iii. The notes justify this as matching `loadMotionEnergy.m` and “interpolates to neural time axis.”

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy uses the same `discretize_continuous(...)` helper: a per-session 50th percentile split into binary low/high categories, with the same zero-threshold fallback and no missing-data category.

ii. ```python
me_disc = discretize_continuous(result['motion_energy'])
...
disc = (v >= threshold).astype(np.float32)
```

iii. The notes explicitly state “Discretize 50th %ile” for motion energy and describe the fixed output values as `['low', 'high']`.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy frame times are corrected by the session video offset and trial go cue and then interpolated onto the same `TIME_AXIS` used for neural data.

ii. ```python
vidshift = find_video_offset(f)
...
aligned_times = frameTimes - vidshift - goCue[trix]
...
me_interp = f_interp(taxis)
```

iii. The notes repeatedly describe motion energy as being “interpolated to neural time axis using video frameTimes.”

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code is permissive and fallback-heavy. It catches many exceptions and silently continues; if motion/kinematic extraction fails it substitutes all-zero time series. Missing `frameTimes` are replaced with a synthetic `np.arange(...)/400.0` clock. Missing aligned samples are nearest-filled for paw and motion energy, and tongue NaN velocities are set to zero. There is no explicit missing/visibility category in the stored outputs.

ii. ```python
except:
    n_frames = ts_data.shape[-1] if ts_data.ndim == 3 else ts_data.shape[0]
    frameTimes = np.arange(1, n_frames + 1) / 400.0
```

```python
if tongue_vel is not None:
    tongue_vel_trials.append(tongue_vel[:, trial_idx].astype(np.float32))
else:
    tongue_vel_trials.append(np.zeros(N_TIMEBINS, dtype=np.float32))
```

```python
xv[np.isnan(xv)] = 0
yv[np.isnan(yv)] = 0
...
me_interp = np.interp(indices, indices[mask], me_interp[mask])
```

iii. The notes justify several of these fallbacks as handling “edge cases,” especially probe indexing, nested motion-energy structs, and the zero-threshold tongue discretization.

## 11-a. What are the most time-consuming steps of the code?

i. The dominant expensive operations in this implementation are the nested neural loops over clusters and trials, and the repeated per-trial interpolation/gradient work for tongue, paw, and motion energy. Unlike the reference implementation, this code does not use a vectorized `histogram2d` spike counter or frame-to-bin reducer.

ii. ```python
for ci, clu_idx in enumerate(cluid):
    ...
    for trial_num in range(1, Ntrials + 1):
        ...
        counts, _ = np.histogram(trial_spikes, bins=EDGES)
```

```python
for trix in range(Ntrials):
    ...
    xpos[:, trix] = fx(taxis)
    ypos[:, trix] = fy(taxis)
```

iii. The notes estimate full processing at roughly 6-7 seconds per session, but do not isolate hotspots. The code structure itself shows where the time is spent.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunities are the per-cluster/per-trial spike histogram loop, the repeated per-trial interpolation loops for tongue/paw/motion-energy alignment, and the repeated per-trial assembly of identical time inputs and repeated categorical output arrays.

ii. ```python
for ci, clu_idx in enumerate(cluid):
    ...
    for trial_num in range(1, Ntrials + 1):
        ...
```

```python
for trix in range(Ntrials):
    ...
    xpos[:, trix] = fx(taxis)
    ypos[:, trix] = fy(taxis)
```

```python
for trial_idx in range(len(result['neural_trials'])):
    time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
    ...
    lick_dir = np.full(N_TIMEBINS, info['lick_direction'], dtype=np.int64)
```

iii. The notes do not acknowledge these vectorization opportunities. The trajectory instead emphasizes correctness fixes and successful decoder training.

## 11-c. What processing does the code repeat multiple times?

i. Several computations are repeated. `find_video_offset(f)` is recomputed independently inside tongue, paw, and motion-energy loaders for the same session. Feature-name extraction is repeated within each movement function. The constant `TIME_AXIS` input is recreated for every trial. When `--show-processing` is used, the continuous outputs are discretized again for plotting after already being discretized for the saved dataset.

ii. ```python
vidshift = find_video_offset(f)
```

This appears in:
```python
def compute_tongue_velocity(...):
    ...
    vidshift = find_video_offset(f)
```
```python
def compute_paw_velocity(...):
    ...
    vidshift = find_video_offset(f)
```
```python
def load_motion_energy(...):
    ...
    vidshift = find_video_offset(f)
```

iii. The notes do not present these as repeated work; they mostly describe the pipeline at a higher level.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads and stores several values that are never used downstream: `L`, absolute spike times `tm_abs` / `spike_tm_abs`, the `all_cluid` list, and the unused `session_idx` argument. It also builds optional plotting-only summaries and repeats trial-level time vectors and repeated categorical outputs rather than storing compact trial scalars.

ii. ```python
L = bp['L'][:].flatten()  # left trials
...
tm_abs_ref = clu_group['tm'][clu_idx, 0]
spike_tm_abs = f[tm_abs_ref][:].flatten()
...
all_cluid = []
...
all_cluid.append(cluid)
```

iii. The notes do not call these out as wasteful. Their focus is on decoder validity rather than code efficiency.
