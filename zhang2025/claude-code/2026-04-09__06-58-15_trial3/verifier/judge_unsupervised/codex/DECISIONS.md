# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads a session manifest from `bwm_release.csv`, groups rows by session `eid`, and then loads each session directly from the local ONE cache on disk instead of using the ONE API. For each session it reads the trials parquet, wheel arrays, camera timestamps and motion-energy arrays, and per-probe spike-sorting outputs from `alf/.../pykilosort`.

ii. 
```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)

session_groups = bwm_df.groupby('eid')
session_list = []
for eid, group in session_groups:
    row = group.iloc[0]
    probe_names = list(group['probe_name'])
    session_list.append((
        eid, row['lab'], row['subject'], row['date'],
        row['session_number'], probe_names
    ))
```

```python
def load_trials(session_path):
    alf_path = os.path.join(session_path, 'alf')
    trials_file = find_file(alf_path, '_ibl_trials.table.pqt')
    if trials_file is None:
        return None
    return pd.read_parquet(trials_file)
```

```python
for pname in probe_names:
    spikes, clusters = load_spike_data(session_path, pname)
    if spikes is not None:
        spikes_list.append(spikes)
        clusters_list.append(clusters)
```

iii. The notes say the agent deliberately used “direct disk loading (bypasses ONE API since cache is read-only)” and tried to mirror the reference loaders while operating on the cached files already present on disk.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column in `bwm_release.csv`. As sessions are processed, each unique subject name is appended to `subjects`, and each session gets a `subject_idx` pointing into that list.

ii.
```python
subject_set = []
...
subject = result['subject']
if subject not in subject_set:
    subject_set.append(subject)
subject_idx = subject_set.index(subject)
```

```python
data = {
    ...
    'subjects': subject_set,
    'subject_idx': np.array(all_subject_idx, dtype=np.int64),
    ...
}
```

iii. The agent’s notes treat the BWM release table as the authoritative session manifest, so subject identities are inherited from that table rather than inferred from filesystem traversal.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in `bwm_release.csv`. All probe rows sharing an `eid` are grouped into one session, and the corresponding probe names are merged during session processing.

ii.
```python
session_groups = bwm_df.groupby('eid')
for eid, group in session_groups:
    row = group.iloc[0]
    probe_names = list(group['probe_name'])
    session_list.append((
        eid, row['lab'], row['subject'], row['date'],
        row['session_number'], probe_names
    ))
```

```python
if len(spikes_list) == 1:
    merged_spikes = spikes_list[0]
    ...
else:
    merged_spikes, cluster_brain_ids = merge_probes(spikes_list, clusters_list)
```

iii. In the notes, the agent states that the reference code merges probes within a session because probes from the same session are not treated as independent.

## 1-d. How are the data split into trials?

i. The script loads the full trials table for a session, computes one alignment interval per row from `stimOn_times`, bins spikes and behaviors over every row, then keeps only rows that survive the combined trial/behavior masks. Each surviving row becomes one trial in the output lists.

ii.
```python
stim_times = trials_df[ALIGN_TIME].values
interval_begs = stim_times + TIME_WINDOW[0]
interval_ends = stim_times + TIME_WINDOW[1]
```

```python
good_indices = np.where(combined_mask)[0]

neural_data = binned_spikes[good_indices]
wheel_data = binned_wheel[good_indices]
me_data = binned_me[good_indices]
```

```python
for trial_idx in range(n_trials):
    neural_trial = neural_data[trial_idx].astype(np.uint8)
    ...
    neural_list.append(neural_trial)
    input_list.append(input_trial)
    output_list.append(output_trial)
```

iii. The notes say this is meant to mirror `load_trials_and_mask()`, `bin_spiking_data()`, and the reference behavior interpolation on a per-trial basis after alignment to stimulus onset.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by a base mask and then by continuous-behavior coverage. The base mask excludes NaNs in key trial fields, reaction times outside 0.08 to 2.0 s, trials longer than 10 s from `goCue_times` to `feedback_times`, and no-choice trials. After that, the script also requires wheel and whisker data to span the aligned window closely enough for interpolation.

ii.
```python
for event in NAN_EXCLUDE:
    if event in trials_df.columns:
        mask &= ~trials_df[event].isna()

rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
mask &= (rt >= MIN_RT)
mask &= (rt <= MAX_RT)

trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()

mask &= (trials_df['choice'] != 0)
```

```python
if np.abs(t_beg - trial_times[0]) > BINSIZE:
    continue
if np.abs(t_end - trial_times[-1]) > BINSIZE:
    continue
...
combined_mask = mask & wheel_mask & me_mask
```

iii. The notes explicitly justify the first mask as matching `load_trials_and_mask()` defaults plus `prepare_data(max_trial_len=10.0)`. The trajectory shows the agent debugged wheel coverage and then kept the same start/end tolerance checks because they matched `get_behavior_per_interval()`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from spike timestamps and cluster assignments in each probe’s spike-sorting output. Brain-region annotations are derived separately from cluster channel assignments and channel brain-location IDs.

ii.
```python
spikes = {
    'times': np.load(times_file).flatten(),
    'clusters': np.load(clusters_file).flatten(),
}
```

```python
channels_file = os.path.join(spike_dir, 'clusters.channels.npy')
...
for candidate_dir in [probe_path, spike_dir]:
    for name in ['electrodeSites.brainLocationIds_ccf_2017.npy',
                  'channels.brainLocationIds_ccf_2017.npy']:
        ...
clusters['brain_ids'] = brain_ids
```

iii. The notes say the agent followed the reference `load_spiking_data()` / `merge_probes()` pathway and treated spike times plus cluster ids as the actual neural signal source.

## 2-b. How is the `neural` data processed?

i. Per session, spikes from all probes are merged into one cluster space, cluster brain IDs are mapped into Beryl acronyms, and spike counts are binned into 20 ms bins over a 2 s window around stimulus onset. The final stored representation is per trial with shape `(n_neurons, 100)` and cast to `uint8`.

ii.
```python
merged_spikes, cluster_brain_ids = merge_probes(spikes_list, clusters_list)
...
cluster_acronyms = br.id2acronym(cluster_brain_ids)
cluster_beryl = br.acronym2acronym(cluster_acronyms, mapping='Beryl')
```

```python
binned_spikes = bin_spikes_fast(
    merged_spikes['times'], merged_spikes['clusters'],
    interval_begs, interval_ends, n_clusters
)
```

```python
neural_trial = neural_data[trial_idx].astype(np.uint8)
```

iii. The notes say this was chosen to match the reference code’s “all neurons, 20 ms bins, stimOn alignment” pipeline while using a faster `np.bincount` implementation for speed and `uint8` for memory.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script does not apply neuron-level QC filtering. It keeps all clusters that appear in the spike files, only dropping whole sessions when spikes are missing, there are zero clusters, or there are fewer than two valid trials after masking.

ii.
```python
# No neuron quality filtering in load_spike_data()
if os.path.exists(metrics_file):
    clusters['metrics'] = pd.read_parquet(metrics_file)
...
n_clusters = int(merged_spikes['clusters'].max()) + 1 if len(merged_spikes['clusters']) > 0 else 0
if n_clusters == 0:
    print(f"  No clusters found for {eid}")
    return None
```

```python
if n_good_trials < 2:
    print(f"  Too few valid trials ({n_good_trials}) for {eid}")
    return None
```

iii. The notes explicitly say “No neuron quality filtering” because the Zhang reference code calls `load_spiking_data(..., qc=None)` even though the data paper also discusses stricter well-isolated-neuron criteria.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`. The script uses a fixed window from 0.5 s before to 1.5 s after stimulus onset and bins spikes within that window.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

```python
stim_times = trials_df[ALIGN_TIME].values
interval_begs = stim_times + TIME_WINDOW[0]
interval_ends = stim_times + TIME_WINDOW[1]
```

```python
b = np.clip(((t - interval_begs[trial_idx]) / BINSIZE).astype(np.int32), 0, N_BINS - 1)
```

iii. The notes justify this as following both the task’s explicit “Temporally align based on stimulus onset” instruction and the reference `0_data_caching.py` parameters.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins and 100 bins per trial. There is no further temporal rebinning beyond counting spikes into those 20 ms bins and interpolating behavior onto the same 100-bin grid.

ii.
```python
BINSIZE = 0.02
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The notes say 20 ms was taken directly from the reference caching code and the methods excerpt stating that the unified cached dataset uses 100 time steps over 2 s.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from `stimOn_times` only indirectly: the aligned trial window is defined relative to stimulus onset, and the input is then a deterministic within-trial time axis for those aligned bins.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
time_since_onset = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The notes describe this as a task-specific derived input rather than a raw data stream, with the bin positions chosen to match the reference behavior interpolation grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The script creates one fixed 100-element vector running from the first bin center after window start to the window end, then copies that vector into every trial. It does not recompute anything from per-trial timestamps once the global bin geometry is set.

ii.
```python
time_since_onset = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
...
input_trial[0, :] = time_since_onset
```

iii. The justification in the notes is that the input should be time-varying and aligned to the same 20 ms grid as the rest of the converted dataset.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned by construction: the same `TIME_WINDOW` and `N_BINS` are used for both spikes and inputs, and the time vector is broadcast into each trial’s input array.

ii.
```python
input_trial = np.zeros((2, N_BINS), dtype=np.float32)
input_trial[0, :] = time_since_onset
input_trial[1, :] = trial_num_in_block[trial_idx]
```

iii. The notes repeatedly emphasize a unified stimulus-onset alignment across neural, input, and output variables.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the per-trial `probabilityLeft` sequence in the trials table.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
trial_num_in_block = all_trial_nums[good_indices].astype(np.float32)
```

iii. The notes explicitly say the agent planned to compute trial number in block “from transitions in probabilityLeft values.”

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The script walks through the full unfiltered `probabilityLeft` vector, resets the counter whenever the value changes, and assigns `i - current_block_start` within each constant-probability run. Afterward it subsets to valid trials and broadcasts the scalar across time bins within each trial.

ii.
```python
def compute_trial_number_in_block(prob_left):
    trial_numbers = np.zeros(len(prob_left), dtype=np.float32)
    current_block_start = 0
    current_val = prob_left[0]

    for i in range(len(prob_left)):
        if prob_left[i] != current_val:
            current_block_start = i
            current_val = prob_left[i]
        trial_numbers[i] = i - current_block_start
```

```python
input_trial[1, :] = trial_num_in_block[trial_idx]
```

iii. The notes justify this as preserving the true block structure of the session, and the code comment says it uses the full `trials_df` “for correct block computation.”

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the trials table.

ii.
```python
choice_raw = trials_df['choice'].values[good_indices]
```

iii. The notes identify `trials.choice` as the source variable and note that IBL encodes it as `-1`, `0`, or `1`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. No-choice trials are removed by the trial mask. The remaining choices are remapped from IBL coding to decoder coding using `-1 -> 0` for left and `1 -> 1` for right, then broadcast across all 100 time bins for each trial.

ii.
```python
mask &= (trials_df['choice'] != 0)
...
choice = ((choice_raw + 1) / 2).astype(np.int32)
...
output_trial[0, :] = choice[trial_idx]
```

iii. The notes explicitly list “Choice mapping: IBL choice -1 (left) -> 0, choice 1 (right) -> 1” to satisfy the task’s requested label convention.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column in the trials table.

ii.
```python
prob_left = trials_df['probabilityLeft'].values[good_indices]
```

iii. The notes say the agent interpreted the task’s “prior probability of left” as the reference code’s `block` variable, namely `probabilityLeft`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The scalar probabilities are discretized by direct value mapping: `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`. The resulting class is then broadcast across the time dimension within a trial.

ii.
```python
prior = np.zeros(len(prob_left), dtype=np.int32)
prior[prob_left == 0.2] = 0
prior[prob_left == 0.5] = 1
prior[prob_left == 0.8] = 2
...
output_trial[1, :] = prior[trial_idx]
```

iii. The notes list this mapping explicitly as a planned variable conversion driven by the decoder-task specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from raw wheel position and wheel timestamp arrays: `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
pos_file = find_file(alf_path, '_ibl_wheel.position.npy')
ts_file = find_file(alf_path, '_ibl_wheel.timestamps.npy')
...
re_pos = np.load(pos_file).flatten()
re_ts = np.load(ts_file).flatten()
```

iii. The trajectory shows the agent originally used a simpler gradient estimate, then revised the code after checking `SessionLoader.load_wheel()` and the wheel helper functions in the reference stack.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The script resamples wheel position to a uniform 1000 Hz grid, applies an order-8 Butterworth low-pass filter with 20 Hz corner, differentiates the filtered trace to get velocity, takes the absolute value to get speed, then linearly interpolates that speed into the 100 stimulus-aligned trial bins.

ii.
```python
fs = 1000
t = np.arange(re_ts[0], re_ts[-1], 1.0 / fs)
position = scipy_interp1d(re_ts, re_pos, kind='linear')(t)
...
sos = scipy.signal.butter(N=order, Wn=corner_frequency / fs * 2,
                           btype='lowpass', output='sos')
vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, position)), 0, 0) * fs
speed = np.abs(vel).astype(np.float32)
```

```python
binned_wheel, wheel_mask = interpolate_behavior(
    wheel_times, wheel_speed, interval_begs, interval_ends)
```

iii. The notes and trajectory justify this as matching `brainbox` / `SessionLoader.load_wheel()` more faithfully than the original draft, specifically after the agent inspected the library code.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. After interpolation, all wheel-speed samples from all valid trials within a session are flattened and discretized into 3 equal-frequency bins using session-specific percentiles. The labels are `0`, `1`, `2` corresponding to `low`, `medium`, `high`.

ii.
```python
wheel_flat = wheel_data.flatten()
wheel_discrete = discretize_to_bins(wheel_flat, n_bins=3)
wheel_discrete_2d = wheel_discrete.reshape(wheel_data.shape).astype(np.int32)
```

```python
quantiles = np.linspace(0, 100, n_bins + 1)
thresholds = np.percentile(valid, quantiles)
result = np.digitize(values, thresholds[1:-1], right=False).astype(np.int32)
```

iii. The notes say this was a task-driven decision: the outputs had to be categorical, and the agent chose equal-frequency bins so the classes would be balanced.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same stimulus-centered intervals used for spikes. For each trial, the wheel trace inside `[stimOn - 0.5 s, stimOn + 1.5 s]` is linearly interpolated onto `N_BINS` sample points from `t_beg + BINSIZE` to `t_end`, and trials are rejected if the wheel trace does not cover the interval closely enough.

ii.
```python
idxs_beg = np.searchsorted(beh_times, interval_begs, side='right')
idxs_end = np.searchsorted(beh_times, interval_ends, side='left')
...
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
y_interp = interp1d(trial_times, trial_vals, kind='linear',
                   fill_value='extrapolate')(x_interp)
```

iii. The notes say this was chosen to mirror `get_behavior_per_interval()` while obeying the task’s stimulus-onset alignment requirement.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from either `leftCamera.ROIMotionEnergy.npy` with `_ibl_leftCamera.times.npy`, or, if those are unavailable or inconsistent, from the corresponding right-camera files.

ii.
```python
me_file = find_file(alf_path, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_path, '_ibl_leftCamera.times.npy')
...
me_file = find_file(alf_path, 'rightCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_path, '_ibl_rightCamera.times.npy')
```

iii. The notes explicitly say “Whisker ME: tries left camera first, falls back to right,” citing the same preference used by the reference code.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Beyond left-first/right-fallback loading, the script does not transform the motion-energy values before trialization. It linearly interpolates the sampled motion-energy signal into the 100 aligned bins for each trial.

ii.
```python
if me_file is not None and times_file is not None:
    me = np.load(me_file).flatten()
    times = np.load(times_file).flatten()
    if len(me) == len(times):
        return times, me
```

```python
binned_me, me_mask = interpolate_behavior(
    me_times, me_values, interval_begs, interval_ends)
```

iii. The notes justify this as matching the reference behavior loader, which takes whisker motion energy directly from the camera-derived signal.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, whisker motion energy is flattened across all valid time points within a session and discretized into 3 quantile bins, then reshaped back to `(n_trials, 100)`.

ii.
```python
me_flat = me_data.flatten()
me_discrete = discretize_to_bins(me_flat, n_bins=3)
me_discrete_2d = me_discrete.reshape(me_data.shape).astype(np.int32)
```

iii. The notes say the rationale was the same as for wheel speed: satisfy the required categorical output format while keeping roughly balanced classes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned exactly like wheel speed: same trial intervals, same interpolation grid, same coverage checks, and same 100-bin output length.

ii.
```python
if me_times is not None:
    binned_me, me_mask = interpolate_behavior(
        me_times, me_values, interval_begs, interval_ends)
else:
    binned_me = np.full((len(trials_df), N_BINS), np.nan, dtype=np.float32)
    me_mask = np.zeros(len(trials_df), dtype=bool)
```

iii. The notes describe a single unified alignment strategy for all converted variables.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or inconsistent files cause the relevant loader to return `None`. Trials with missing required events or insufficient behavior coverage are masked out. Sessions are skipped entirely if trials tables or spike files are missing, if no clusters remain, or if fewer than two valid trials survive. Missing brain-region IDs fall back to `'void'`.

ii.
```python
if trials_file is None:
    return None
...
if pos_file is None or ts_file is None:
    return None, None
...
if len(re_pos) != len(re_ts) or len(re_pos) < 2:
    return None, None
```

```python
if len(spikes_list) == 0:
    print(f"  No spike data found for {eid}")
    return None
...
if n_good_trials < 2:
    print(f"  Too few valid trials ({n_good_trials}) for {eid}")
    return None
```

```python
if cluster_brain_ids is not None:
    ...
else:
    cluster_beryl = np.array(['void'] * n_clusters)
```

iii. The notes describe a “skip on error / skip on no valid trials” policy and explicitly explain the final loss of 21 sessions and 4 subjects as a consequence of those masks.

## 10-a. What are the most time-consuming steps of the code?

i. According to the notes, the dominant costs are loading spikes, binning spikes, and loading/processing wheel data for each session, followed by reassembly/pickling for the full dataset.

ii.
```python
binned_spikes = bin_spikes_fast(...)
...
wheel_times, wheel_speed = load_wheel_speed(session_path)
...
for batch_start in range(0, len(session_meta), BATCH_SIZE):
    ...
    with open(session_meta[idx]['tmp_file'], 'rb') as f:
        sess = pickle.load(f)
```

iii. The notes contain explicit runtime estimates: about 0.9 s to load spikes, 0.4 s to bin spikes, and 0.5 s to load wheel data per session, with full conversion dominated by those stages.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious remaining/vectorizable loops are the unused spike-by-spike loop in `bin_spikes_vectorized`, the per-trial loop in `bin_spikes_fast`, the per-neuron loop that builds `brain_region_idx`, and the per-trial formatting loop that constructs Python lists of arrays.

ii.
```python
for spike_i in range(len(trial_times)):
    c = trial_clusters[spike_i]
    b = bin_idx[spike_i]
    if c < n_clusters_total:
        binned[trial_idx, c, b] += 1
```

```python
for i, trial_idx in enumerate(valid_idx):
    ...
    counts = np.bincount(lin_idx, minlength=minlength)
    binned[trial_idx] = counts[:minlength].reshape(n_clusters_total, N_BINS)
```

```python
for neuron_idx in range(result['n_neurons']):
    region = result['cluster_beryl'][neuron_idx]
    ...
for trial_idx in range(n_trials):
    ...
```

iii. The notes mention the agent already replaced a slower `np.add.at` style approach with `np.bincount`, but they also imply speed and memory remained central constraints, so these remaining Python loops are the natural next targets.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly searches revision directories via `find_file`, repeatedly performs `searchsorted`-based interval extraction for both wheel and whisker traces, and reads every temporary session pickle twice during the final reassembly pass.

ii.
```python
for d in sorted(os.listdir(base_path), reverse=True):
    if d.startswith('#') and d.endswith('#'):
        revisions.append(d)
...
for rev in revisions:
    candidate = os.path.join(base_path, rev, filename)
```

```python
idxs_beg = np.searchsorted(beh_times, interval_begs, side='right')
idxs_end = np.searchsorted(beh_times, interval_ends, side='left')
```

```python
for meta in session_meta:
    with open(meta['tmp_file'], 'rb') as f:
        sess = pickle.load(f)
...
for batch_start in range(0, len(session_meta), BATCH_SIZE):
    ...
    with open(session_meta[idx]['tmp_file'], 'rb') as f:
        sess = pickle.load(f)
```

iii. This is not justified in detail in the notes, but the memory-fix notes explicitly say the agent switched to temp files and then reassembled them later, which necessarily introduced repeated reading.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads cluster metrics, channels, and depths although the decoder dataset only uses region labels; it computes continuous wheel and whisker traces only to discard them after discretization; and it expands scalar per-trial variables (`choice`, `prior`, `trial_number_in_block`) across all 100 time bins even though they are constant within a trial.

ii.
```python
if os.path.exists(channels_file):
    clusters['channels'] = np.load(channels_file).flatten()
if os.path.exists(depths_file):
    clusters['depths'] = np.load(depths_file).flatten()
if os.path.exists(metrics_file):
    clusters['metrics'] = pd.read_parquet(metrics_file)
```

```python
wheel_data = binned_wheel[good_indices]
me_data = binned_me[good_indices]
...
wheel_discrete_2d = wheel_discrete.reshape(wheel_data.shape).astype(np.int32)
me_discrete_2d = me_discrete.reshape(me_data.shape).astype(np.int32)
```

```python
output_trial[0, :] = choice[trial_idx]
output_trial[1, :] = prior[trial_idx]
input_trial[1, :] = trial_num_in_block[trial_idx]
```

iii. The memory notes show the agent was aware of output size pressure, but these extra continuous intermediates and repeated broadcasts were kept anyway because they simplified matching the target format.
