# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script does not discover sessions dynamically. It hard-codes the 25 fixed-delay `Ephys_Behavior` sessions in `SESSION_META`, then opens each `data_structure_<animal>_<date>.mat` file with `h5py`. Motion energy is loaded separately from `motionEnergy_<animal>_<date>.mat`. This means the agent intentionally restricted loading to the fixed-delay two-context ALM dataset, not the randomized-delay dataset.

ii.
```python
DATA_DIR = 'data/Ephys_Behavior'

SESSION_META = [
    ('EKH1', '2021-08-07', [1]),
    ('EKH3', '2021-08-11', [1, 2]),
    ('JEB6', '2021-04-18', [2]),
    ...
    ('JGR3', '2021-11-18', [1]),
]

def process_session(animal, date, probes, show_processing=False, session_idx=0):
    session_id = f'{animal}_{date}'
    data_file = os.path.join(DATA_DIR, f'data_structure_{session_id}.mat')
    me_file = os.path.join(DATA_DIR, f'motionEnergy_{session_id}.mat')
    ...
    f = h5py.File(data_file, 'r')
```

iii. In `CONVERSION_NOTES.md` Step 5 the agent justified this as “Use Ephys_Behavior only: Has DR+WC two-context task.” The trajectory shows it matched those 25 sessions to the reference `load*_ALMVideo.m` scripts and treated that as the intended dataset.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `animal` field of each processed session. After processing, the script collects unique animal IDs with `sorted(set(...))`, stores them in `subjects`, and writes one `subject_idx` per session.

ii.
```python
result = {
    ...
    'session_id': session_id,
    'animal': animal,
}

subjects = sorted(list(set(r['animal'] for r in all_results)))

for result in all_results:
    subj_idx = subjects.index(result['animal'])
    subject_idx_list.append(subj_idx)
```

iii. The notes explicitly acknowledge a discrepancy: “Subjects | 10 animals loaded | paper counts 9 | Using all 10.” The agent chose to trust the files and loading scripts over the paper’s summary count.

## 1-c. How are the data split into sessions?

i. Each entry in `SESSION_META` defines one session. `main()` iterates over those tuples, calls `process_session(...)`, and appends one session-level result to `all_results`, which later becomes one element in each top-level `neural`, `input`, and `output` list.

ii.
```python
for idx, (animal, date, probes) in enumerate(sessions):
    result = process_session(animal, date, probes,
                           show_processing=args.show_processing,
                           session_idx=idx)
    if result is not None:
        all_results.append(result)

for result in all_results:
    neural_list.append(result['neural_trials'])
    ...
    input_list.append(session_inputs)
    output_list.append(session_outputs)
```

iii. The agent’s notes say session membership came from the MATLAB animal-specific loading scripts, including the dual-probe EKH3 session.

## 1-d. How are the data split into trials?

i. Within each session, the code reads session-wide behavioral arrays from `obj.bp`, builds a boolean `valid_mask`, finds valid trial indices, and then packages per-trial neural and behavioral arrays only for those indices. Trial splitting is therefore index-based off the raw Bpod arrays.

ii.
```python
bp = f['obj']['bp']
Ntrials = int(bp['Ntrials'][0, 0])

L = bp['L'][:].flatten()
R = bp['R'][:].flatten()
hit = bp['hit'][:].flatten()
miss = bp['miss'][:].flatten()
no = bp['no'][:].flatten()
autowater = bp['autowater'][:].flatten()
early = bp['early'][:].flatten()
stim_enable = bp['stim']['enable'][:].flatten()

valid_mask = (early == 0) & (no == 0) & (stim_enable == 0) & ((hit == 1) | (miss == 1))
valid_trials = np.where(valid_mask)[0]

for trial_idx in valid_trials:
    neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32))
```

iii. The notes say “Include hit and miss trials” and “Exclude early, no, stim trials.” The trajectory shows this was a deliberate decoder-oriented choice.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered with one boolean rule: remove early-lick trials, ignore/no-response trials, and stimulation trials; keep hit and miss trials. Sessions with fewer than 2 remaining trials are skipped entirely.

ii.
```python
valid_mask = (early == 0) & (no == 0) & (stim_enable == 0) & ((hit == 1) | (miss == 1))
valid_trials = np.where(valid_mask)[0]

if len(valid_trials) < 2:
    print(f'    Skipping {session_id}: only {len(valid_trials)} valid trials')
    f.close()
    return None
```

iii. In `CONVERSION_NOTES.md` the agent cited the paper’s exclusion of early and ignore trials and the MATLAB conditions’ exclusion of stimulation. It added miss trials so `outcome` could be decoded.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from the cluster structures under `obj.clu`. The code uses cluster quality strings plus the per-spike `trialtm` and `trial` arrays; `goCue` is used only for alignment.

ii.
```python
clu_ref = f['obj']['clu'][probe_idx, 0]
clu_group = f[clu_ref]
qualities = get_cluster_qualities(f, clu_group)

tm_ref = clu_group['trialtm'][clu_idx, 0]
trial_ref = clu_group['trial'][clu_idx, 0]

spike_trialtm = f[tm_ref][:].flatten()
spike_trial = f[trial_ref][:].flatten().astype(int)
goCue = bp['ev']['goCue'][:].flatten()
```

iii. The notes map “Spike times (clu)” to `neural` and describe the reference chain as `alignSpikes -> getSeq -> removeLowFRClusters`.

## 2-b. How is the `neural` data processed?

i. For each kept cluster, spikes are aligned to go cue, histogrammed into 10 ms bins from -2.5 s to 2.5 s, converted to firing rate, and smoothed with a 15-sample causal Gaussian kernel with reflect padding. Dual-probe sessions are concatenated across neurons.

ii.
```python
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2

counts, _ = np.histogram(trial_spikes, bins=EDGES)
fr = counts / PARAMS['dt']
fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
trialdat[:, ci, trial_num - 1] = fr_smooth

trialdat = np.concatenate(all_trialdat, axis=1)
```

iii. The notes say this was intended to match MATLAB `getSeq.m` and `mySmooth.m`: 10 ms bins, causal Gaussian smoothing, reflect boundary, firing-rate units.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural QC has three stages. First, drop clusters whose quality string is one of `garbage`, `gabrga`, `noisy`, or `real?`. Second, drop neurons with mean firing rate `<= 1 Hz`. Third, skip sessions with fewer than 10 neurons after filtering. The FR screen is computed from `trialdat` averaged over all trials and time, not from condition-averaged PSTHs as in the reference MATLAB.

ii.
```python
PARAMS = {
    ...
    'lowFR': 1.0,
    'quality_exclude': ['garbage', 'gabrga', 'noisy', 'real?'],
}

cluid = find_clusters(qualities, PARAMS['quality_exclude'])

meanFRs = np.nanmean(np.nanmean(trialdat, axis=2), axis=0)
fr_mask = meanFRs > PARAMS['lowFR']
trialdat = trialdat[:, fr_mask, :]

if n_neurons < 10:
    print(f'    Skipping {session_id}: only {n_neurons} neurons (need >= 10)')
    ...
```

iii. The notes justify 1 Hz from the paper and the quality list from `findClusters.m`. The trajectory also shows the agent fixed an early bug so empty/null quality strings are included, matching MATLAB `quality='all'`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to go-cue onset by subtracting the trial’s `goCue` time from each spike’s within-trial time, then binning the aligned spike times on the fixed `[-2.5, 2.5]` window.

ii.
```python
goCue = bp['ev']['goCue'][:].flatten()
...
spike_goCue = goCue[spike_trial_valid - 1]
spike_aligned = spike_trialtm_valid - spike_goCue
...
counts, _ = np.histogram(trial_spikes, bins=EDGES)
```

iii. The notes and trajectory explicitly cite MATLAB `alignSpikes.m` and the instruction “Temporally align based on Go cue onset.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins (`dt = 1/100`) over 500 time bins from -2.5 s to 2.5 s. No later temporal rebinning is applied.

ii.
```python
PARAMS = {
    ...
    'dt': 1/100,
}

EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
N_TIMEBINS = len(TIME_AXIS)
```

iii. The notes say the agent chose 10 ms because most reference scripts use `1/100`, even though `getDefaultParams.m` defaults to `1/200` in other contexts.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This input is not read as a raw time series. It is derived from the chosen alignment event (`goCue`) plus the fixed analysis window and bin size.

ii.
```python
PARAMS = {
    'alignEvent': 'goCue',
    'tmin': -2.5,
    'tmax': 2.5,
    'dt': 1/100,
}

TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
...
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
```

iii. The notes describe `input[0]` simply as “Time from goCue in seconds” and treat it as a derived time basis shared with the neural bins.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code computes bin centers from the fixed edge vector, casts them to `float32`, reshapes to `(1, n_timepoints)`, and reuses the same vector for every trial.

ii.
```python
EDGES = np.arange(PARAMS['tmin'], PARAMS['tmax'] + PARAMS['dt'], PARAMS['dt'])
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
...
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
session_inputs.append(time_input)
```

iii. There is no separate justification beyond matching the neural time axis and the decoder specification.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is exactly co-registered with neural data because it reuses the same `TIME_AXIS` used for neural binning. Each trial receives the identical aligned vector.

ii.
```python
TIME_AXIS = EDGES[:-1] + PARAMS['dt'] / 2
...
fr_smooth = causal_gaussian_smooth(fr, PARAMS['smooth'], PARAMS['bctype'])
...
time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
```

iii. The notes repeatedly describe all streams as aligned to go cue on the same 10 ms grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived directly from the behavioral `R` array in `obj.bp`; `R=1` is right and `R=0` implies left.

ii.
```python
R = bp['R'][:].flatten()
...
lick_direction = R.copy()
```

iii. The notes map `R (right trial)` to `output[0]` with “left=0, right=1.”

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The script just copies the right-trial indicator and, during output packaging, broadcasts it across all time bins for each kept trial.

ii.
```python
lick_direction = R.copy()
...
lick_dir = np.full(N_TIMEBINS, info['lick_direction'], dtype=np.int64)
...
output_array = np.stack([
    lick_dir,
    context,
    outcome,
    ...
], axis=0)
```

iii. The agent did not give a deeper justification; it treated lick direction as a direct behavioral label from the raw trial metadata.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `autowater` in `obj.bp`, where autowater trials correspond to WC and non-autowater trials correspond to DR.

ii.
```python
autowater = bp['autowater'][:].flatten()
...
context = 1 - autowater
```

iii. The notes explicitly say `autowater=1 means WC`, so the conversion used `WC=0, DR=1`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The only transformation is binary inversion: `1 - autowater`. As with lick direction, the per-trial value is then repeated across all time bins.

ii.
```python
context = 1 - autowater
...
context = np.full(N_TIMEBINS, info['context'], dtype=np.int64)
```

iii. This follows the mapping documented in Step 5 of the notes.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `hit` array in `obj.bp`, after earlier trial filtering has already removed ignore and early trials.

ii.
```python
hit = bp['hit'][:].flatten()
miss = bp['miss'][:].flatten()
...
valid_mask = (early == 0) & (no == 0) & (stim_enable == 0) & ((hit == 1) | (miss == 1))
...
outcome = hit.copy()
```

iii. The notes state the mapping as `hit -> correct=1`, with misses serving as incorrect trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code copies `hit`, interprets `1` as correct and `0` as incorrect, and broadcasts that per-trial label across time bins.

ii.
```python
outcome = hit.copy()
...
outcome = np.full(N_TIMEBINS, info['outcome'], dtype=np.int64)
```

iii. The agent’s rationale was mainly decoder-driven: keeping both hit and miss trials makes `outcome` decodable.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-view DLC trajectories in `obj.traj[0]`, specifically the `tongue` feature’s x/y coordinates, plus video `frameTimes`, dropped-frame metadata, the neural-video offset, and `goCue`.

ii.
```python
traj_ref = f['obj']['traj'][0, 0]
traj_group = f[traj_ref]
...
feat_ref = traj_group['featNames'][0, 0]
...
if name == 'tongue':
    tongue_idx = i
...
ft_ref = traj_group['frameTimes'][trix, 0]
frameTimes = f[ft_ref][:].flatten()
...
aligned_times = frameTimes - vidshift - goCue[trix]
```

iii. The notes say this was chosen to match `findPosition + findVelocity` using “side view (view 1), tongue feature.”

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the script interpolates tongue x/y positions from video time to the neural time axis, takes temporal gradients, replaces NaN tongue velocities with zero, and converts x/y velocity into speed magnitude.

ii.
```python
fx = interp1d(aligned_times[valid], x[valid], bounds_error=False, fill_value=np.nan)
fy = interp1d(aligned_times[valid], y[valid], bounds_error=False, fill_value=np.nan)
xpos[:, trix] = fx(taxis)
ypos[:, trix] = fy(taxis)
...
xv = np.gradient(xpos[:, trix])
yv = np.gradient(ypos[:, trix])
xv[np.isnan(xv)] = 0
yv[np.isnan(yv)] = 0
...
speed = np.sqrt(xvel**2 + yvel**2)
```

iii. The notes explicitly cite the MATLAB tongue rule from `findVelocity.m`: tongue NaNs become zero because the tongue is often not visible.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The code nominally uses the per-session 50th percentile across all tongue-speed samples, but if that threshold is exactly zero it switches to the 50th percentile of strictly positive values. The binary label is then `v >= threshold`.

ii.
```python
threshold = np.percentile(all_values, threshold_percentile)

if threshold == 0:
    pos_values = all_values[all_values > 0]
    if len(pos_values) > 0:
        threshold = np.percentile(pos_values, threshold_percentile)

disc = (v >= threshold).astype(np.float32)
```

iii. The trajectory shows the agent added this after noticing that a literal median over mostly-zero tongue speeds produced all-ones labels. The notes describe it as a “Discretization edge case.”

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue trajectories are aligned by subtracting both the neural-video offset and trial-specific `goCue` time from each frame time, then interpolating onto `TIME_AXIS`.

ii.
```python
taxis = TIME_AXIS + PARAMS['advance_movement']
vidshift = find_video_offset(f)
...
aligned_times = frameTimes - vidshift - goCue[trix]
...
xpos[:, trix] = fx(taxis)
ypos[:, trix] = fy(taxis)
```

iii. The notes say this was meant to match MATLAB `findPosition.m` with `advance_movement = 0`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the top-view DLC trajectories in `obj.traj[1]`. The code looks for all feature names containing `paw`, which in these files are `top_paw` and `bottom_paw`.

ii.
```python
traj_ref = f['obj']['traj'][1, 0]
traj_group = f[traj_ref]
...
for i, name in enumerate(feat_names):
    if 'paw' in name.lower():
        paw_indices.append(i)
```

iii. The notes say the agent chose “top view (view 2), top_paw + bottom_paw averaged.”

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw feature is interpolated to the neural time axis, missing positions are filled with nearest-neighbor interpolation, gradients are computed, a baseline derivative is subtracted, speed magnitude is computed for each paw, and the final paw velocity is the mean speed across paw features.

ii.
```python
xp = fx(taxis)
yp = fy(taxis)
...
xp = np.interp(indices, indices[mask], xp[mask])
...
xv = np.gradient(xpos[:, trix])
yv = np.gradient(ypos[:, trix])
basederiv_x = np.nanmedian(np.diff(xpos[:, trix]))
basederiv_y = np.nanmedian(np.diff(ypos[:, trix]))
if not np.isnan(basederiv_x):
    xv = xv - basederiv_x
if not np.isnan(basederiv_y):
    yv = yv - basederiv_y
...
speed = np.sqrt(xvel**2 + yvel**2)
all_speeds.append(speed)
...
avg_speed = np.mean(all_speeds, axis=0)
```

iii. The notes justify this as following `findVelocity.m` for non-tongue features, but the choice to average `top_paw` and `bottom_paw` is an agent-added reduction to satisfy the single-output decoder format.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity uses the same `discretize_continuous(...)` helper as tongue velocity: session-wide median threshold, except a zero threshold is replaced by the median of positive values.

ii.
```python
tongue_disc = discretize_continuous(result['tongue_vel'])
paw_disc = discretize_continuous(result['paw_vel'])
me_disc = discretize_continuous(result['motion_energy'])
```

iii. There is no separate paw-specific justification; the agent applied the same thresholding policy to all continuous outputs.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw positions are aligned exactly like tongue positions: `frameTimes - vidshift - goCue`, then interpolation to `TIME_AXIS`.

ii.
```python
taxis = TIME_AXIS + PARAMS['advance_movement']
vidshift = find_video_offset(f)
...
aligned_times = frameTimes - vidshift - goCue[trix]
...
xp = fx(taxis)
yp = fy(taxis)
```

iii. The notes say the movement streams were meant to use the same alignment as the reference kinematic helpers, with no temporal shift.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<session>.mat` file, specifically the `me.data` series for each trial, together with video `frameTimes` from `obj.traj` and `goCue` times from behavior.

ii.
```python
me_raw = sio.loadmat(me_file)
me_struct = me_raw['me'][0, 0]
me_data = me_struct['data']
...
traj_ref = f['obj']['traj'][0, 0]
traj_group = f[traj_ref]
ft_ref = traj_group['frameTimes'][trix, 0]
frameTimes = f[ft_ref][:].flatten()
```

iii. The notes cite MATLAB `loadMotionEnergy.m` and specifically mention the nested `me.data.data` case for JEB15 sessions.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code loads `me.data`, unwraps nested structs when necessary, aligns each trial to go cue on the neural time axis by interpolation, and fills NaNs with nearest values.

ii.
```python
me_struct = me_raw['me'][0, 0]
me_data = me_struct['data']
if me_data.dtype.names is not None and 'data' in me_data.dtype.names:
    me_data = me_data[0, 0]['data']
...
aligned_times = frameTimes - vidshift - goCue[trix]
...
f_interp = interp1d(aligned_times[valid], me_trial[valid],
                   bounds_error=False, fill_value=np.nan)
me_interp = f_interp(taxis)
...
me_interp = np.interp(indices, indices[mask], me_interp[mask])
```

iii. The agent explicitly added the nested-struct unwrapping after finding it in trajectory step 83 and matching the comment in `loadMotionEnergy.m`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded with the same shared helper as the other continuous outputs: session-wise median, except that a zero threshold is replaced with the median of positive values.

ii.
```python
me_disc = discretize_continuous(result['motion_energy'])
...
threshold = np.percentile(all_values, threshold_percentile)
if threshold == 0:
    pos_values = all_values[all_values > 0]
    if len(pos_values) > 0:
        threshold = np.percentile(pos_values, threshold_percentile)
```

iii. The notes do not provide a separate justification beyond using a common discretizer for all continuous outputs.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Each motion-energy trace is aligned by subtracting the neural-video offset and trial `goCue` from frame times, then interpolating to the 10 ms neural time grid.

ii.
```python
taxis = TIME_AXIS + PARAMS['advance_movement']
vidshift = find_video_offset(f)
...
aligned_times = frameTimes - vidshift - goCue[trix]
...
me_interp = f_interp(taxis)
```

iii. The notes say this was intended to reproduce MATLAB `loadMotionEnergy.m` with `advance_movement = 0`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed data are handled permissively. The code uses many broad `try/except` blocks; if frame times are missing it fabricates `1/400 s` frame times, if interpolated arrays contain NaNs it fills them with nearest values, if tongue velocity is NaN it becomes zero, and if an entire movement stream fails for a session it replaces that output with zeros for every valid trial.

ii.
```python
try:
    ft_ref = traj_group['frameTimes'][trix, 0]
    frameTimes = f[ft_ref][:].flatten()
except:
    n_frames = ts_data.shape[-1] if ts_data.ndim == 3 else ts_data.shape[0]
    frameTimes = np.arange(1, n_frames + 1) / 400.0

...
xv[np.isnan(xv)] = 0
yv[np.isnan(yv)] = 0

...
if motion_energy is not None:
    me_trials.append(motion_energy[:, trial_idx].astype(np.float32))
else:
    me_trials.append(np.zeros(N_TIMEBINS, dtype=np.float32))
```

iii. The notes frame these as “edge case” handling. The trajectory shows the agent preferred keeping sessions running over failing loudly on partial data issues.

## 11-a. What are the most time-consuming steps of the code?

i. The most expensive work is repeated per-neuron, per-trial spike histogramming and smoothing, followed by per-trial interpolation for tongue, paw, and motion-energy traces. The conversion log shows full processing took about 189 s for 25 sessions.

ii.
```python
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

iii. The notes’ timing tables and `conversion_full_out.txt` identify session processing as roughly 4-16 s each, with the heaviest sessions being those with many neurons and trials.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The clear vectorization targets are the nested cluster-by-trial spike loop, the per-column smoothing loop inside `causal_gaussian_smooth`, the repeated per-trial interpolation loops for tongue/paw/motion energy, and the repeated per-trial packaging loops used to build outputs.

ii.
```python
for j in range(x_padded.shape[1]):
    result[:, j] = convolve(x_padded[:, j], kern, mode='same')

for ci, clu_idx in enumerate(cluid):
    ...
    for trial_num in range(1, Ntrials + 1):
        ...

for trix in range(Ntrials):
    ...
```

iii. The agent mentions efficiency goals in Step 6 of the instructions, but the final script remains mostly loop-based.

## 11-c. What processing does the code repeat multiple times?

i. The code repeatedly recomputes `vidshift`, repeatedly parses feature names from HDF5 references, separately interpolates x/y for each feature and trial, discretizes the same continuous outputs again inside plotting, and rebuilds the same time input for every trial.

ii.
```python
vidshift = find_video_offset(f)
...
feat_ref = traj_group['featNames'][0, 0]
feat_data = f[feat_ref]
feat_names = []
for j in range(feat_data.shape[1]):
    ...

tongue_disc = discretize_continuous(tongue)
paw_disc = discretize_continuous(paw)
me_disc = discretize_continuous(me)

time_input = TIME_AXIS.astype(np.float32).reshape(1, -1)
session_inputs.append(time_input)
```

iii. The notes do not call these out explicitly, but they are evident from the implementation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads absolute spike times `tm` but never uses them, computes continuous tongue/paw/motion-energy arrays only to discard them after binarization, prepares `trialdat_full`, `valid_trials`, and other debug-only fields in `show_processing` mode, and processes neural data for all trials before dropping invalid trials at packaging time.

ii.
```python
tm_abs_ref = clu_group['tm'][clu_idx, 0]
spike_tm_abs = f[tm_abs_ref][:].flatten()

result['trialdat_full'] = trialdat
result['valid_trials'] = valid_trials
result['goCue'] = goCue

for trial_num in range(1, Ntrials + 1):
    ...
    trialdat[:, ci, trial_num - 1] = fr_smooth
...
for trial_idx in valid_trials:
    neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32))
```

iii. These were not justified in the notes; they appear to be convenience/debug leftovers rather than requirements of the downstream decoder.
