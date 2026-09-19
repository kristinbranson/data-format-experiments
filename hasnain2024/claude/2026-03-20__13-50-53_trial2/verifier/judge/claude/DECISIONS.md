# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes 44 session definitions in `SESSION_DEFS` as tuples of (animal, date, probes, data_dir_key). Each session is loaded from a MATLAB `.mat` file (`data_structure_<anm>_<date>.mat`). Two loaders handle v7.3 HDF5 and v5 formats. Motion energy is loaded separately from `motionEnergy_<anm>_<date>.mat`. The sessions are split across two directories: `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior`.

ii.
```python
SESSION_DEFS = [
    ('EKH1', '2021-08-07', [2], 'ephys'),
    ...
    ('JEB24', '2023-11-03', [1], 'random'),
]

for i, (anm, date, probes, ddir) in enumerate(session_defs):
    result = load_session(anm, date, probes, ddir, PARAMS)
```

Loading raw data (HDF5 path):
```python
def _load_raw_data_hdf5(data_fn, probes):
    f = h5py.File(data_fn, 'r')
    obj = f['obj']
    ...
```

iii. The AI identified the 44 sessions from the authors' loading scripts (`load<ANM>_ALMVideo.m`), excluding sessions without data files (JEB4, JEB5) and data files without loading scripts. This is documented in CONVERSION_NOTES.md Steps 1-4.

## 1-b. How are the data split into subjects?

i. The animal name is taken directly from the first element of each `SESSION_DEFS` tuple. Subject list is built as the sorted set of unique animal names. `subject_idx` maps each session to its subject.

ii.
```python
all_animals = sorted(set(s['animal'] for s in all_sessions))
animal_to_idx = {a: i for i, a in enumerate(all_animals)}
...
'subjects': all_animals,
'subject_idx': np.array(subject_idx, dtype=np.int64),
```

iii. The animal ID is explicit in the session definition tuples.

## 1-c. How are the data split into sessions?

i. Each entry in `SESSION_DEFS` defines one session. The `data_dir_key` field (`'ephys'` or `'random'`) selects which subfolder the file is in. Each session becomes one element of the `neural`, `input`, and `output` lists. The result is 44 sessions: 25 from Ephys_Behavior and 19 from RandomizedDelay_Ephys_Behavior.

ii.
```python
DATA_DIRS = {
    'ephys': os.path.join(DATA_ROOT, 'Ephys_Behavior'),
    'random': os.path.join(DATA_ROOT, 'RandomizedDelay_Ephys_Behavior'),
}
...
data_fn = os.path.join(data_dir, f'data_structure_{anm}_{date}.mat')
```

iii. Sessions correspond to individual `.mat` files as organized by the authors.

## 1-d. How are the data split into trials?

i. Trials are defined by the `Ntrials` field in `obj.bp`. All per-trial arrays are truncated to this length. Valid trials are those that pass the early-lick and photostim filters. Spike data carries explicit trial assignments via `clu.trial`.

ii.
```python
bp['Ntrials'] = int(bp_group['Ntrials'][()].flat[0])
ntrials = bp['Ntrials']
for field in ['hit', 'miss', 'no', 'early', 'L', 'R', 'autowater']:
    bp[field] = bp_group[field][()].flatten().astype(np.float64)[:ntrials]
```

iii. The Bpod structure defines trials directly with `Ntrials`, and all fields are indexed by trial number.

## 1-e. How are trials filtered based on quality controls?

i. Two filters: early-lick trials (`bp.early == 1`) and photostimulation trials (`bp.stim.enable == 1`) are removed. Unlike the reference, the AI does **not** filter out trials that extend past the end of the recording. This leads to all-zero neural data for 61 trials across 2 sessions (sessions 36 and 43).

ii.
```python
trial_mask = (bp['early'] == 0) & (bp['stim_enable'] == 0)
valid_trials = np.where(trial_mask)[0] + 1  # 1-indexed
```

iii. The AI documented the early-lick and photostim exclusion in CONVERSION_NOTES.md, citing the reference code. The all-zero trials were noticed during validation but not filtered out, documented as "acceptable (0.44% of all trials)".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Spike-sorted clusters from `obj.clu{probe}`, specifically the `trialtm` (spike times relative to trial start), `trial` (which trial each spike belongs to), and `quality` (curation label). Go cue times `bp.ev.goCue` provide the alignment event.

ii.
```python
spike_tm = all_spike_times[clu_idx]
spike_trial = all_spike_trials[clu_idx]
...
spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
counts, _ = np.histogram(spk_times, bins=edges)
```

iii. Same source variables as the reference pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are binned at **10 ms** (dt=1/100), converted to firing rates (counts/dt), then smoothed with a **causal** Gaussian kernel of window size 15. The causal kernel zeros out the first half of the Gaussian window, matching `mySmooth.m` from the reference code. Boundary conditions use reflection padding.

ii.
```python
PARAMS = {
    'dt': 1.0 / 100,  # 10 ms bins
    'smooth_window': 15,
    ...
}

def causal_gaussian_smooth(x, N, bctype='reflect'):
    kern = np.array(sig_windows.gaussian(N, std=(N - 1) / (2 * 2.5)))
    kern[:N // 2] = 0  # Make causal
    kern = kern / kern.sum()
    ...
    out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
```

Spike binning and smoothing:
```python
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
...
fr = causal_gaussian_smooth(counts.astype(np.float64) / params['dt'],
                           params['smooth_window'],
                           params['smooth_bctype'])
```

iii. The AI chose 10 ms bins citing `WorkingWithDataObjs.m` tutorial (`params.dt = 1/100`) rather than `getDefaultParams.m` (`params.dt = 1/200` = 5 ms). The causal Gaussian smoothing matches the reference `mySmooth.m` implementation. The AI noted this choice in CONVERSION_NOTES.md Step 5.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Cluster quality labels are lower-cased and checked against the exclude set `{'garbage', 'noisy', 'gabrga', 'real?'}`. Additionally, empty labels and 'nan' are excluded. (2) Units with mean firing rate <= 1 Hz are removed. Sessions with fewer than 10 surviving units are skipped entirely.

ii.
```python
quality_exclude = params['quality_exclude']  # {'garbage', 'noisy', 'gabrga', 'real?'}
keep_mask = np.array([q not in quality_exclude and q != '' and q != 'nan'
                      for q in all_qualities])
...
mean_fr = np.mean(trialdat, axis=(1, 2))
fr_mask = mean_fr > params['low_fr']
```

iii. The quality exclusion set matches the reference `findClusters.m`. The 1 Hz threshold matches the paper. The min 10 units per session filter matches the paper's inclusion criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting the go cue time of the corresponding trial: `spike_time - align_times[trial_num - 1]`. The aligned spike times are then binned into the time window [-2.5, 2.5] s.

ii.
```python
align_times = ev[params['align_event']]  # 'goCue'
...
spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
counts, _ = np.histogram(spk_times, bins=edges)
```

iii. This matches the reference `alignSpikes.m` approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses **10 ms** bins (dt = 1/100), producing 500 time bins spanning [-2.5, 2.5] s. No rebinning is applied; spikes are binned directly at this resolution.

ii.
```python
PARAMS = {
    'dt': 1.0 / 100,  # 10 ms bins
    ...
}
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
time_axis = edges[:-1] + params['dt'] / 2
n_timebins = len(time_axis)  # 500
```

iii. The AI cited `WorkingWithDataObjs.m` as the source for 10 ms bins. The reference solution uses 5 ms bins (1000 time points) from `getDefaultParams.m` (`params.dt = 1/200`).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the time axis definition itself -- it is the bin centers of the time grid spanning [-2.5, 2.5] s, which is defined by the go cue alignment.

ii.
```python
time_axis = edges[:-1] + params['dt'] / 2
...
time_input = time_axis.astype(np.float32).reshape(1, -1)
sess_input.append(time_input)
```

iii. The time axis is constructed from the binning parameters, not from any raw data variable.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing -- the time axis is the bin centers of the neural data grid, defined analytically.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is the neural binning grid itself. Bin centers are computed from the same edges used for spike histogramming, so the k-th time input value corresponds exactly to the k-th neural time bin.

ii.
```python
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
time_axis = edges[:-1] + params['dt'] / 2
...
counts, _ = np.histogram(spk_times, bins=edges)
```

iii. Alignment is exact by construction.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI uses `bp.R` directly as the lick direction variable, where R=1 means right and R=0 means left.

ii.
```python
lick_direction = bp['R'][valid_trial_indices].copy()
```

iii. The AI treated lick direction as binary (left=0, right=1), using `bp.R` directly rather than deriving it from the combination of hit/miss and instructed side.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI directly copies `bp.R` as the lick direction, producing a binary output: 0=left, 1=right. There is no "no lick" class. Ignore trials (where the animal didn't lick) are assigned the instructed side rather than a separate class.

ii.
```python
lick_direction = bp['R'][valid_trial_indices].copy()
...
lick_dir = np.full((1, n_timebins), int(sess['lick_direction'][t]), dtype=np.int64)
```

Output values:
```python
'output_values': [
    ['left', 'right'],        # lick_direction
    ...
]
```

iii. The AI's approach conflates the instructed direction with the actual lick direction. On ignore trials the animal did not lick, but bp.R still indicates the instructed side. On miss trials, bp.R is the instructed side but the animal licked the opposite direction. The reference solution derives actual lick direction from hit/miss+R and includes a "no lick" class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The AI uses `bp.autowater`, where autowater=1 indicates water-cued (WC) context and autowater=0 indicates delayed-response (DR) context.

ii.
```python
behavioral_context = 1.0 - bp['autowater'][valid_trial_indices]
```

iii. This correctly uses the `autowater` field to distinguish contexts.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The value is computed as `1 - autowater`, so WC (autowater=1) maps to 0 and DR (autowater=0) maps to 1. This produces a binary output matching the instruction's WC=0, DR=1 encoding.

ii.
```python
behavioral_context = 1.0 - bp['autowater'][valid_trial_indices]
...
context = np.full((1, n_timebins), int(sess['behavioral_context'][t]), dtype=np.int64)
```

Output values:
```python
['WC', 'DR'],  # behavioral_context
```

iii. The mapping is correct: WC=0, DR=1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI uses `bp.hit` directly. Hit=1 maps to correct=1, otherwise (miss or ignore) maps to incorrect=0.

ii.
```python
outcome = bp['hit'][valid_trial_indices].copy()
```

iii. Only `bp.hit` is used, not `bp.miss` or `bp.no`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The outcome is binary: correct (1) for hit trials, incorrect (0) for all others (both miss and ignore). The instructions specify three classes: incorrect, correct, and ignore -- but the AI only implements two classes.

ii.
```python
outcome = bp['hit'][valid_trial_indices].copy()
...
outc = np.full((1, n_timebins), int(sess['outcome'][t]), dtype=np.int64)
```

Output values:
```python
['incorrect', 'correct'],  # outcome
```

iii. The AI merged ignore trials into the incorrect class, losing the three-class structure specified in the instructions.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI uses DLC tracking data from `obj.traj`, specifically the `top_tongue` feature from the bottom camera (view index 2, adjusted to 1 in code). It extracts x, y coordinates and frame times.

ii.
```python
tongue_vel = _compute_feature_velocity_generic(
    traj_data, ntrials, view=2, feat_name='top_tongue',
    vidshift=vidshift, align_times=align_times,
    time_axis=time_axis, is_tongue=True)
```

iii. The AI uses only one camera view (bottom) for tongue tracking, while the reference uses both side (`tongue`) and bottom (`top_tongue`) cameras, normalizing and averaging them.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI interpolates x, y positions to the neural time axis using linear interpolation (`interp1d`), then computes velocity using `np.gradient`. For tongue specifically, NaN velocities are set to 0. Speed is computed as `sqrt(xvel^2 + yvel^2)`. The result is discretized at the 50th percentile into 2 classes: low (0) and high (1). NaN entries map to 0 (low).

ii.
```python
xpos = interp1d(ft_a[valid], x_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
ypos = interp1d(ft_a[valid], y_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
...
if is_tongue:
    xvel[np.isnan(xvel)] = 0
    yvel[np.isnan(yvel)] = 0
speed[:, trix] = np.sqrt(xvel**2 + yvel**2).astype(np.float32)
```

Discretization:
```python
def _discretize_velocity(data, n_timebins, n_trials):
    threshold = np.percentile(valid_vals, 50)
    disc = (data >= threshold).astype(np.int64)
    disc[np.isnan(data)] = 0  # Replace NaN with 0
    return disc
```

iii. The AI's approach differs from the reference in several ways: (1) interpolation to grid vs. frame-resolution velocity then binning, (2) NaN velocity set to 0 instead of a "not visible" class, (3) no per-run smoothing of tracked coordinates, (4) only one camera view, (5) 2 output classes instead of 3.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Two classes split at the 50th percentile of non-NaN values across the session. NaN bins are assigned class 0 (low) rather than a separate "not visible" class.

ii.
```python
threshold = np.percentile(valid_vals, 50)
disc = (data >= threshold).astype(np.int64)
disc[np.isnan(data)] = 0
```

Output values: `['low', 'high']`

iii. The instructions specify three classes: 0 (< 50th percentile), 1 (>= 50th percentile), 2 (not visible). The AI only implements two classes, mapping "not visible" to the "low" class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by subtracting the video offset (computed as median difference between SpikeGLX bitcode and behavioral bitStart) and the trial's go cue time. The corrected positions are then interpolated to the neural time axis using `interp1d`.

ii.
```python
ft_aligned = ft - vidshift - align_times[trix]
...
xpos = interp1d(ft_a[valid], x_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
```

Video offset:
```python
bit_start = np.nanmedian(bp_group['ev']['bitStart'][()].flatten())
bitcode_bitstart = np.nanmedian(sglx['bitcode']['bitstart'][()].flatten())
vidshift = bitcode_bitstart / fs - bit_start
```

iii. The video offset computation uses `nanmedian` rather than `mode` as in the reference. Interpolation to the neural grid differs from the reference approach of binning frame-resolution values.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The AI uses `top_paw` from the bottom camera (view index 2), same DLC tracking data in `obj.traj`.

ii.
```python
paw_vel = _compute_feature_velocity_generic(
    traj_data, ntrials, view=2, feat_name='top_paw',
    vidshift=vidshift, align_times=align_times,
    time_axis=time_axis, is_tongue=False)
```

iii. Same source as the reference (bottom camera, `top_paw`).

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Similar to tongue but with different NaN handling: x, y positions are interpolated to the neural time axis, NaN values are filled with nearest valid values (`_fill_nearest`), then velocity is computed via `np.gradient`. A baseline velocity (median of the velocity) is subtracted. Speed is `sqrt(xvel^2 + yvel^2)`. Discretized at 50th percentile into 2 classes.

ii.
```python
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

iii. The baseline subtraction and nearest-neighbor filling are unique to the AI's implementation. The reference does not apply baseline subtraction or nearest-fill -- it preserves NaN for untracked frames and uses a "not visible" class.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: 2 classes at 50th percentile, NaN mapped to 0.

ii.
```python
paw_vel_disc = _discretize_velocity(paw_vel, n_timebins, n_valid_trials)
```

iii. Missing the "not visible" (class 2) specified in the instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same approach as tongue: frame times corrected by video offset and go cue, then positions interpolated to the neural time axis.

ii. Same as 7-d.

iii. Same as 7-d.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from separate `motionEnergy_<anm>_<date>.mat` files. The motion energy trace has one value per camera frame per trial.

ii.
```python
def _load_motion_energy_generic(me_fn, traj_data, vidshift, align_times, time_axis, ntrials):
    me_file = sio.loadmat(me_fn, squeeze_me=True)
    me_struct = me_file['me']
    me_cell = me_struct['data'].item()
```

iii. Same source as the reference.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values are interpolated to the neural time axis using `interp1d`, then NaN values are filled with nearest valid values. Discretized at the 50th percentile into 2 classes.

ii.
```python
me_aligned[:, trix] = interp1d(ft_aligned[valid], me_trial[valid],
                               kind='linear', bounds_error=False,
                               fill_value=np.nan)(time_axis).astype(np.float32)
...
for trix in range(ntrials):
    col = me_aligned[:, trix]
    if not np.isnan(col).all() and np.isnan(col).any():
        me_aligned[:, trix] = _fill_nearest(col)
```

iii. The reference simply bins (averages) frame values into bins without interpolation and preserves NaN for empty bins. The AI's interpolation and nearest-fill approach fabricates values in bins where no measurement exists.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same 2-class discretization at 50th percentile. Sessions where motion energy loading failed get all-zero values (all "low").

ii.
```python
me_disc = _discretize_velocity(me_data, n_timebins, n_valid_trials)
```

iii. Missing the "no video" class (class 2) specified in the instructions. 4 sessions with unreadable ME files silently get all-zero values rather than a "no video" flag.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from the side camera are corrected by video offset and go cue time. Motion energy values are then interpolated to the neural time axis using linear interpolation.

ii.
```python
ft_aligned = ft - vidshift - align_times[trix]
me_aligned[:, trix] = interp1d(ft_aligned[valid], me_trial[valid],
                               kind='linear', bounds_error=False,
                               fill_value=np.nan)(time_axis)
```

iii. The reference uses binning (averaging frames into bins) rather than interpolation.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches: (1) For missing frame times, the AI fabricates times assuming 400 Hz: `ft = np.arange(1, me_trial.size + 1) / 400.0`. (2) For NaN coordinates (untracked frames), the AI fills with nearest valid values for paw, and sets velocity to 0 for tongue. (3) For unreadable motion energy files, the session gets all-zero motion energy. (4) Trials past the recording end are kept with all-zero neural data. (5) Generic try/except blocks silently skip problematic trials or data.

ii.
```python
if ft is None or ft.size == 0 or np.all(np.isnan(ft)):
    ft = np.arange(1, me_trial.size + 1) / 400.0
...
if not is_tongue:
    xpos = _fill_nearest(xpos)
...
if is_tongue:
    xvel[np.isnan(xvel)] = 0
```

iii. The reference preserves missing data as NaN and uses a "not visible" class rather than fabricating values. The broad try/except blocks in the AI code could mask real errors.

## 11-a. What are the most time-consuming steps of the code?

i. Loading and spike binning. The per-neuron, per-trial spike binning loop is the dominant cost. The full conversion takes ~163 seconds for 44 sessions.

ii.
```python
for neuron_idx, clu_idx in enumerate(good_indices):
    for t_idx, trial_num in enumerate(valid_trials):
        spk_mask = spike_trial == trial_num
        spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
        counts, _ = np.histogram(spk_times, bins=edges)
        fr = causal_gaussian_smooth(counts.astype(np.float64) / params['dt'], ...)
```

iii. The nested loop over neurons and trials for spike binning is the main bottleneck.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates over every neuron and every trial individually, calling `np.histogram` once per (neuron, trial) pair. This could be vectorized using `np.histogram2d` (as the reference does) to bin all trials at once for each neuron. The smoothing is also applied per-neuron per-trial rather than in a batch.

ii.
```python
for neuron_idx, clu_idx in enumerate(good_indices):
    for t_idx, trial_num in enumerate(valid_trials):
        spk_mask = spike_trial == trial_num
        ...
        counts, _ = np.histogram(spk_times, bins=edges)
        fr = causal_gaussian_smooth(counts / params['dt'], ...)
        trialdat[neuron_idx, :, t_idx] = fr
```

iii. The reference uses `np.histogram2d` to bin all trials simultaneously for each cluster, which is significantly more efficient.

## 11-c. What processing does the code repeat multiple times?

i. The spike masking `spike_trial == trial_num` is computed for every trial inside every neuron loop, meaning each neuron's spike array is scanned N_trials times. Frame time correction (subtracting vidshift and align_times) is computed per-trial for velocity and again for motion energy. The causal Gaussian kernel is reconstructed inside `causal_gaussian_smooth` on every call rather than being built once.

ii.
```python
# Called for every neuron x trial combination:
spk_mask = spike_trial == trial_num
...
# Kernel rebuilt on every call:
kern = np.array(sig_windows.gaussian(N, std=(N - 1) / (2 * 2.5)))
```

iii. The reference pre-builds constants at module level and uses vectorized operations to avoid repeated computation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The raw velocity arrays (`tongue_vel_raw`, `paw_vel_raw`, `me_raw`) are computed and stored in the session result dict but not included in the final output pickle. (2) The loading functions read all fields from `obj.bp` including `no`, `L`, `sample`, `delay` which are not used in the final output. (3) Baseline velocity subtraction for paw is computed but its effect is lost after discretization.

ii.
```python
# Raw velocities stored but not saved:
'tongue_vel_raw': tongue_vel,
'paw_vel_raw': paw_vel,
'me_raw': me_data,
```

iii. The raw velocities are used for processing plots but are not part of the final converted dataset.
