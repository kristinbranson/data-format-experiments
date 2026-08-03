# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB file `data_structure_<anm>_<date>.mat`, located in one of two directories (`Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`). The 44 sessions and their probes are hard-coded in `SESSION_DEFS`. The file format is detected (HDF5 vs v5), and a separate loader is used for each format (`_load_raw_data_hdf5` and `_load_raw_data_v5`). Motion energy is loaded from a separate `motionEnergy_<anm>_<date>.mat` file. Each session is loaded once via `load_session`.

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

Loading detects format:
```python
fmt = _detect_file_format(data_fn)
if fmt == 'hdf5':
    raw = _load_raw_data_hdf5(data_fn, probes)
else:
    raw = _load_raw_data_v5(data_fn, probes)
```

iii. From CONVERSION_NOTES.md: The sessions are derived from the authors' loading scripts. Sessions missing from data (JEB4, JEB5) are excluded, and data files without loading scripts are excluded. Both MATLAB formats are handled.

## 1-b. How are the data split into subjects?

i. The animal name is passed as a separate field from `SESSION_DEFS` (the first element of each tuple). Subjects are collected as a sorted set of unique animal names, and `subject_idx` maps each session to its subject index.

ii.
```python
SESSION_DEFS = [
    ('EKH1', '2021-08-07', [2], 'ephys'),
    ...
]

all_animals = sorted(set(s['animal'] for s in all_sessions))
animal_to_idx = {a: i for i, a in enumerate(all_animals)}
```

iii. The animal name is directly available from the session definition tuples.

## 1-c. How are the data split into sessions?

i. One session is one entry in `SESSION_DEFS`, keyed by `(animal, date)`. Each becomes one element of the neural, input, and output lists. Sessions from both task types (fixed-delay and randomized-delay) are included. The `data_dir_key` field maps each session to its folder.

ii.
```python
data_dir = DATA_DIRS[data_dir_key]
data_fn = os.path.join(data_dir, f'data_structure_{anm}_{date}.mat')
```

iii. CONVERSION_NOTES.md documents 44 sessions: 25 fixed-delay and 19 randomized-delay.

## 1-d. How are the data split into trials?

i. Trials are defined by the `Ntrials` field in `obj.bp`. Per-trial fields (`hit`, `miss`, `early`, `R`, `L`, `autowater`, `stim_enable`, event times) are truncated to `Ntrials` entries. Spike data carries trial assignments per spike.

ii.
```python
bp['Ntrials'] = int(bp_group['Ntrials'][()].flat[0])
ntrials = bp['Ntrials']
for field in ['hit', 'miss', 'no', 'early', 'L', 'R', 'autowater']:
    bp[field] = bp_group[field][()].flatten().astype(np.float64)[:ntrials]
```

iii. The Bpod table defines trials directly and there is one go cue per trial.

## 1-e. How are trials filtered based on quality controls?

i. Two filters are applied: early-lick trials (`bp.early == 1`) and photostimulation trials (`bp.stim.enable == 1`) are dropped. Unlike the reference, trials past the end of the neural recording are **not** explicitly filtered.

ii.
```python
trial_mask = (bp['early'] == 0) & (bp['stim_enable'] == 0)
valid_trials = np.where(trial_mask)[0] + 1  # 1-indexed
```

iii. CONVERSION_NOTES.md documents: "Exclude early lick and stim/photoinactivation trials (~early, ~stim.enable)". No mention of filtering trials past recording end.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The spike-sorted clusters from `obj.clu`, specifically each cluster's `trialtm` (spike times relative to trial start), `trial` (trial assignment per spike), and `quality` (curation label). Go cue times `bp.ev.goCue` provide alignment.

ii.
```python
trialtm = unit['trialtm'].flatten().astype(np.float64)
trial = unit['trial'].flatten().astype(np.float64)
...
spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
counts, _ = np.histogram(spk_times, bins=edges)
```

iii. Same variables as the reference pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into **10 ms** bins (dt=1/100) spanning -2.5 to 2.5 s (500 bins). Counts are divided by `dt` to get firing rates in Hz. The rates are smoothed with a **causal** Gaussian kernel of window size 15, implemented via `causal_gaussian_smooth` which replicates `mySmooth.m`. The causal kernel zeros out the first half of the Gaussian window.

ii.
```python
PARAMS = {
    'dt': 1.0 / 100,  # 10 ms bins
    'smooth_window': 15,
    ...
}

edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
counts, _ = np.histogram(spk_times, bins=edges)
fr = causal_gaussian_smooth(counts.astype(np.float64) / params['dt'],
                           params['smooth_window'],
                           params['smooth_bctype'])
```

Causal smoothing:
```python
def causal_gaussian_smooth(x, N, bctype='reflect'):
    kern = np.array(sig_windows.gaussian(N, std=(N - 1) / (2 * 2.5)))
    kern[:N // 2] = 0  # Make causal
    kern = kern / kern.sum()
    ...
```

iii. CONVERSION_NOTES.md: "dt=1/100 (10ms bins)", "Causal Gaussian kernel, window=15". The AI chose 10ms bins based on `WorkingWithDataObjs.m` tutorial parameter `dt=1/100`, and causal smoothing based on `mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters. (1) Cluster quality: labels matching `{'garbage', 'noisy', 'gabrga', 'real?'}` are excluded, as well as empty or 'nan' labels. (2) Firing rate: units with mean FR <= 1 Hz are removed. (3) Minimum units: sessions with fewer than 10 units after filtering are skipped entirely.

ii.
```python
PARAMS = {
    'quality_exclude': {'garbage', 'noisy', 'gabrga', 'real?'},
    'low_fr': 1.0,
    'min_units': 10,
}

keep_mask = np.array([q not in quality_exclude and q != '' and q != 'nan'
                      for q in all_qualities])
...
mean_fr = np.mean(trialdat, axis=(1, 2))
fr_mask = mean_fr > params['low_fr']
...
if n_neurons < params['min_units']:
    return None
```

iii. CONVERSION_NOTES.md: "Quality filter: Exclude 'garbage', 'noisy', 'gabrga', 'real?'" and "sessions included for analysis only if they had at least 10 units". The label `'poor'` is NOT in the exclusion list (unlike the reference which drops it).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting the go cue time for each trial: `trialtm - goCue[trial]`. This is done per-neuron, per-trial in a nested loop.

ii.
```python
align_times = ev[params['align_event']]  # 'goCue'
...
spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
counts, _ = np.histogram(spk_times, bins=edges)
```

iii. Aligns to goCue as specified in the instructions and paper.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is **10 ms** (dt = 1/100), producing **500** time bins spanning -2.5 to 2.5 s from the go cue. No rebinning is applied. This differs from the reference which uses 5 ms bins (dt = 1/200, 1000 bins).

ii.
```python
'dt': 1.0 / 100,  # 10 ms bins
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
time_axis = edges[:-1] + params['dt'] / 2
```

iii. CONVERSION_NOTES.md justification: "Neural data time bin: 10ms (dt=1/100) from WorkingWithDataObjs.m params". The AI used the tutorial's `dt=1/100` rather than `getDefaultParams.m`'s `dt=1/200`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The time axis is defined by the binning parameters (tmin, tmax, dt). It is the center of each time bin, not derived from any raw data variable.

ii.
```python
time_axis = edges[:-1] + params['dt'] / 2
...
time_input = time_axis.astype(np.float32).reshape(1, -1)
```

iii. Same approach as reference: the time axis is defined by the binning grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing needed. The time axis is simply the bin centers of the neural binning grid.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the neural binning grid itself. Spike times are aligned to the go cue and counted into the bin edges, and the input is the center of those same bins.

ii.
```python
edges = np.arange(params['tmin'], params['tmax'] + params['dt'], params['dt'])
time_axis = edges[:-1] + params['dt'] / 2
```

iii. Same approach as reference.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The per-trial field `obj.bp.R` is used directly. A value of 1 means right-instructed, 0 means left-instructed. Unlike the reference, `hit` and `miss` are NOT used to infer actual lick direction.

ii.
```python
lick_direction = bp['R'][valid_trial_indices].copy()
```

iii. CONVERSION_NOTES.md mapping: "obj.bp.R/L -> output[0]: lick_direction, L=0, R=1, per-trial".

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. `bp.R` is used directly as the lick direction: right=1, left=0. This gives the **instructed** direction, not the **actual** lick direction. There is no third class for "no lick" (ignore trials where the animal didn't respond). The output has only 2 classes: `['left', 'right']`.

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

iii. The AI treats this as the instructed direction rather than deriving the actual lick from hit/miss + R/L. No "no lick" class is provided.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The per-trial field `obj.bp.autowater`. Autowater=1 indicates WC (water-cued) context.

ii.
```python
bp['autowater'] = bp_raw['autowater'][0, 0].flatten().astype(np.float64)[:ntrials]
...
behavioral_context = 1.0 - bp['autowater'][valid_trial_indices]
```

iii. Same source variable as reference.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. `1 - autowater`: autowater=1 maps to WC=0, autowater=0 maps to DR=1. This produces the correct encoding (WC=0, DR=1).

ii.
```python
behavioral_context = 1.0 - bp['autowater'][valid_trial_indices]
```

iii. Matches the instruction's WC=0, DR=1 encoding.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The per-trial field `obj.bp.hit` is used directly. Only hit (correct=1) vs not-hit (incorrect=0) is distinguished. `bp.miss` is not used to separate misses from ignores.

ii.
```python
outcome = bp['hit'][valid_trial_indices].copy()
```

iii. CONVERSION_NOTES.md: "hit=1 -> correct=1, miss=1 -> incorrect=0".

## 6-b. What processing is involved in computing `output` *Outcome*?

i. `bp.hit` is used directly: hit=1 becomes correct=1, everything else (miss + ignore) becomes incorrect=0. There is no third "ignore" class. The output has only 2 classes: `['incorrect', 'correct']`.

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

iii. The AI maps both miss and ignore trials to "incorrect" (0), collapsing three categories into two.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DLC tracking in `obj.traj`, specifically the `top_tongue` feature from the bottom camera (view index 1). Only ONE camera view is used. The reference uses both cameras (side `tongue` and bottom `top_tongue`).

ii.
```python
tongue_vel = _compute_feature_velocity_generic(
    traj_data, ntrials, view=2, feat_name='top_tongue',
    vidshift=vidshift, align_times=align_times,
    time_axis=time_axis, is_tongue=True)
```

iii. CONVERSION_NOTES.md: "Tongue velocity: Compute from bottom cam DLC features (top_tongue or similar)."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The x and y positions are interpolated to the neural time axis using `interp1d`. Then `np.gradient` computes velocity on the interpolated data. For tongue, NaN velocities are set to 0. Speed is `sqrt(xvel^2 + yvel^2)`. No likelihood filtering, no per-run smoothing, no normalization across cameras.

ii.
```python
xpos = interp1d(ft_a[valid], x_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)
ypos = interp1d(ft_a[valid], y_r[valid], kind='linear',
                bounds_error=False, fill_value=np.nan)(time_axis)

xvel = np.gradient(xpos)
yvel = np.gradient(ypos)

if is_tongue:
    xvel[np.isnan(xvel)] = 0
    yvel[np.isnan(yvel)] = 0

speed[:, trix] = np.sqrt(xvel**2 + yvel**2).astype(np.float32)
```

iii. The AI interpolates positions to the neural time axis before computing velocity, rather than computing velocity at the camera's frame rate and then binning. NaN filtering replaces missing values with 0 velocity rather than marking as "not visible".

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Two classes split at the 50th percentile of non-NaN values. NaN entries (untracked bins) are mapped to class 0 ("low"). There is no third "not visible" class.

ii.
```python
def _discretize_velocity(data, n_timebins, n_trials):
    valid_vals = data[~np.isnan(data)]
    threshold = np.percentile(valid_vals, 50)
    disc = (data >= threshold).astype(np.int64)
    disc[np.isnan(data)] = 0  # NaN -> 0 ("low")
    return disc
```

Output values:
```python
['low', 'high'],  # tongue_velocity
```

iii. The AI uses only 2 classes. Untracked bins are assigned to "low" (0) rather than a separate "not visible" class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by the video offset (`vidshift`) and aligned to the go cue: `ft - vidshift - align_times[trial]`. The video offset is computed as `nanmedian(bitcode_bitstart/fs) - nanmedian(ev.bitStart)`. Positions are then interpolated directly onto the neural time axis.

ii.
```python
bit_start = np.nanmedian(bp_group['ev']['bitStart'][()].flatten())
bitcode_bitstart = np.nanmedian(sglx['bitcode']['bitstart'][()].flatten())
vid_file_offset = bitcode_bitstart / fs
vidshift = vid_file_offset - bit_start
...
ft_aligned = ft - vidshift - align_times[trix]
xpos = interp1d(ft_aligned[valid], x_r[valid], ...)(time_axis)
```

iii. The AI uses `nanmedian` for the offset computation, while the reference uses `mode`. Both implement the same concept from `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DLC tracking `top_paw` from the bottom camera (view index 1).

ii.
```python
paw_vel = _compute_feature_velocity_generic(
    traj_data, ntrials, view=2, feat_name='top_paw',
    vidshift=vidshift, align_times=align_times,
    time_axis=time_axis, is_tongue=False)
```

iii. Same feature as reference (`top_paw` from bottom camera).

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Positions are interpolated to the neural time axis, NaN-filled with nearest-neighbor interpolation, then gradient is computed. A baseline velocity (median of differences) is subtracted. Speed is `sqrt(xvel^2 + yvel^2)`.

ii.
```python
if not is_tongue:
    xpos = _fill_nearest(xpos)
    ypos = _fill_nearest(ypos)

xvel = np.gradient(xpos)
yvel = np.gradient(ypos)

if not is_tongue:
    base_xvel = np.nanmedian(np.diff(xpos))
    base_yvel = np.nanmedian(np.diff(ypos))
    xvel -= (base_xvel if not np.isnan(base_xvel) else 0)
    yvel -= (base_yvel if not np.isnan(base_yvel) else 0)
    xvel = _fill_nearest(xvel)
    yvel = _fill_nearest(yvel)
```

iii. The AI adds baseline subtraction and nearest-neighbor filling for paw (but not tongue), which differs from the reference's approach of computing velocity within contiguous runs of valid frames.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: two classes at 50th percentile, NaN mapped to 0 ("low"). No "not visible" class.

ii.
```python
paw_vel_disc = _discretize_velocity(paw_vel, n_timebins, n_valid_trials)
```

iii. Same approach as tongue velocity discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment as tongue: frame times corrected by vidshift and go cue, then interpolated onto neural time axis.

ii. Same code path as tongue (see 7-d).

iii. Same alignment approach.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. A separate `motionEnergy_<anm>_<date>.mat` file. The motion energy data is loaded and unwrapped from its container structure.

ii.
```python
me_fn = os.path.join(data_dir, f'motionEnergy_{anm}_{date}.mat')
...
me_struct = me_file['me']
me_cell = me_struct['data'].item()
```

iii. Same source as reference.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values are interpolated to the neural time axis using `interp1d`, then NaN values are filled with nearest-neighbor interpolation. This differs from the reference which bins by averaging frames in each bin.

ii.
```python
me_aligned[:, trix] = interp1d(ft_aligned[valid], me_trial[valid],
                               kind='linear', bounds_error=False,
                               fill_value=np.nan)(time_axis).astype(np.float32)
...
# Fill NaN with nearest
me_aligned[:, trix] = _fill_nearest(col)
```

iii. The AI interpolates rather than averaging frames per bin. It also fills remaining NaN values with nearest-neighbor interpolation.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as tongue and paw: two classes at 50th percentile, NaN mapped to 0 ("low").

ii.
```python
me_disc = _discretize_velocity(me_data, n_timebins, n_valid_trials)
```

iii. Same discretization approach as other continuous outputs.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from the side camera are corrected by vidshift and go cue, then motion energy is interpolated onto the neural time axis.

ii.
```python
ft_aligned = ft - vidshift - align_times[trix]
me_aligned[:, trix] = interp1d(ft_aligned[valid], me_trial[valid],
                               kind='linear', bounds_error=False,
                               fill_value=np.nan)(time_axis)
```

iii. Uses interpolation rather than binning.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Missing frame times: if `frameTimes` is empty or all NaN, a synthetic time axis at 400 Hz is generated. (2) Missing motion energy files: if the file can't be loaded, all-zero motion energy is used. (3) NaN in interpolated data: nearest-neighbor fill is applied for paw and motion energy; for tongue, NaN velocities are set to 0. (4) Sessions that fail to load are skipped with a warning.

ii.
```python
if ft is None or ft.size == 0 or np.all(np.isnan(ft)):
    ft = np.arange(1, me_trial.size + 1) / 400.0
...
def _fill_nearest(arr):
    valid_idx = np.where(~nans)[0]
    nan_idx = np.where(nans)[0]
    nearest = np.searchsorted(valid_idx, nan_idx).clip(0, len(valid_idx) - 1)
    arr[nans] = arr[valid_idx[nearest]]
    return arr
```

iii. CONVERSION_NOTES.md documents: 4 JEB23 sessions have corrupted ME files; some sessions have late trials with all-zero neural data. The AI chose to fill NaN values rather than mark them as a separate class.

## 11-a. What are the most time-consuming steps of the code?

i. The spike binning loop is the most time-consuming step. Spikes are binned per-neuron, per-trial in a nested Python loop, which is inherently slow. File loading is also significant. Total conversion runs in ~163 seconds.

ii.
```python
for neuron_idx, clu_idx in enumerate(good_indices):
    for t_idx, trial_num in enumerate(valid_trials):
        spk_mask = spike_trial == trial_num
        spk_times = spike_tm[spk_mask] - align_times[trial_num - 1]
        counts, _ = np.histogram(spk_times, bins=edges)
        fr = causal_gaussian_smooth(counts / params['dt'], ...)
```

iii. The nested loop over neurons and trials for spike binning is the main bottleneck.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning uses a nested loop over neurons and trials (line 460-473), calling `np.histogram` once per neuron per trial. This could be vectorized using `np.histogram2d` over all trials at once per neuron, as the reference does. The velocity computation also loops over trials.

ii.
```python
for neuron_idx, clu_idx in enumerate(good_indices):
    for t_idx, trial_num in enumerate(valid_trials):
        spk_mask = spike_trial == trial_num
        ...
        counts, _ = np.histogram(spk_times, bins=edges)
```

iii. The reference uses a single `histogram2d` call per neuron over all trials, avoiding the inner trial loop.

## 11-c. What processing does the code repeat multiple times?

i. The spike masking `spike_trial == trial_num` is computed for every neuron for every trial. The trial mask could be precomputed once. The smoothing kernel is rebuilt for every neuron-trial combination via `causal_gaussian_smooth`.

ii.
```python
for neuron_idx, clu_idx in enumerate(good_indices):
    for t_idx, trial_num in enumerate(valid_trials):
        spk_mask = spike_trial == trial_num  # recomputed for each neuron
        ...
        fr = causal_gaussian_smooth(...)  # kernel rebuilt each call
```

iii. The reference builds the Gaussian kernel once and applies it with `gaussian_filter1d`.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The full `obj` structure is loaded including fields never used (e.g., `spkWavs`, `tm`, unused tracked features). (2) Nearest-neighbor filling of NaN values in motion energy and paw velocity is done but then the values are discretized, so exact filled values don't matter much. (3) Baseline subtraction in paw velocity computation adds processing that the reference doesn't do and may not be needed.

ii.
```python
# Unnecessary baseline subtraction for paw
base_xvel = np.nanmedian(np.diff(xpos))
base_yvel = np.nanmedian(np.diff(ypos))
xvel -= (base_xvel if not np.isnan(base_xvel) else 0)
yvel -= (base_yvel if not np.isnan(base_yvel) else 0)
```

iii. The baseline subtraction is not in the reference code and is not described in the paper.
