# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not use the ONE API. It loads a release table from `/app/code/code_zhang2025/data/bwm_release.csv`, groups rows by `eid`, reconstructs filesystem paths under `/app/data/one_cache/{lab}/Subjects/{subject}/{date}/{session}/alf`, and then opens parquet and `.npy` files directly. Revisioned files are resolved by choosing the lexicographically latest `#...#` directory.

ii.
```python
DATA_ROOT = '/app/data/one_cache'
BWM_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
```

```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
session_groups = bwm_df.groupby('eid')
for eid, group in session_groups:
    row = group.iloc[0]
    probe_names = list(group['probe_name'])
```

```python
session_path = os.path.join(DATA_ROOT, lab, 'Subjects', subject, date, session_num_str)
trials_df = load_trials(session_path)
spikes, clusters = load_spike_data(session_path, pname)
wheel_times, wheel_speed = load_wheel_speed(session_path)
me_times, me_values = load_whisker_motion_energy(session_path)
```

iii. In `CONVERSION_NOTES.md`, the agent justified direct disk access as a way to “bypass ONE API since cache is read-only,” while still claiming it was matching the reference processing.

## 1-b. How are the data split into subjects?

i. Subjects come from the `subject` column of the release CSV. During assembly the agent builds `subjects` in first-seen order and stores each session’s `subject_idx` as the index into that running list.

ii.
```python
session_list.append((
    eid, row['lab'], row['subject'], row['date'],
    row['session_number'], probe_names
))
```

```python
subject_set = []
...
subject = result['subject']
if subject not in subject_set:
    subject_set.append(subject)
subject_idx = subject_set.index(subject)
```

iii. The notes say the dataset metadata already contains subject identity, so no derivation from file names or behavior is needed.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values in `bwm_release.csv`. The agent groups the release table by `eid`, treating each group as one session and collecting all listed probes into that session.

ii.
```python
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

iii. The notes treat the release table as the authoritative session index.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the rows of `_ibl_trials.table.pqt`. After loading the trials DataFrame, the agent keeps the surviving row indices and slices neural and behavioral arrays by those trial indices.

ii.
```python
def load_trials(session_path):
    alf_path = os.path.join(session_path, 'alf')
    trials_file = find_file(alf_path, '_ibl_trials.table.pqt')
    if trials_file is None:
        return None
    return pd.read_parquet(trials_file)
```

```python
mask = create_trials_mask(trials_df)
combined_mask = mask & wheel_mask & me_mask
good_indices = np.where(combined_mask)[0]
```

iii. The notes say the trials table is already organized one row per trial, so no further trial parsing is necessary.

## 1-e. How are trials filtered based on quality controls?

i. The agent applies a boolean mask requiring non-NaN values in six trial columns, reaction time between `0.08` and `2.0` s, trial length `feedback_times - goCue_times <= 10.0` s when `goCue_times` exists, and `choice != 0`. It then further requires wheel-speed and whisker-motion traces to interpolate successfully over the whole trial window, via `wheel_mask` and `me_mask`.

ii.
```python
NAN_EXCLUDE = [
    'stimOn_times', 'choice', 'feedback_times',
    'probabilityLeft', 'firstMovement_times', 'feedbackType'
]
MIN_RT = 0.08
MAX_RT = 2.0
MAX_TRIAL_LEN = 10.0
EXCLUDE_NOCHOICE = True
```

```python
for event in NAN_EXCLUDE:
    if event in trials_df.columns:
        mask &= ~trials_df[event].isna()
...
rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
mask &= (rt >= MIN_RT)
...
mask &= (rt <= MAX_RT)
...
trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()
...
mask &= (trials_df['choice'] != 0)
```

```python
combined_mask = mask & wheel_mask & me_mask
```

iii. The notes justify this as matching `load_trials_and_mask()` from the Zhang code and combining it with continuous-behavior availability checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural tensor is derived from `spikes.times.npy` and `spikes.clusters.npy` from each probe. Additional cluster/channel arrays are only used to annotate regions, not to generate the actual neural activity values.

ii.
```python
spikes = {
    'times': np.load(times_file).flatten(),
    'clusters': np.load(clusters_file).flatten(),
}
```

```python
binned_spikes = bin_spikes_fast(
    merged_spikes['times'], merged_spikes['clusters'],
    interval_begs, interval_ends, n_clusters
)
```

iii. The notes describe this as following the reference’s spike-time plus cluster-identity representation.

## 2-b. How is the `neural` data processed?

i. For each trial window, the agent bins spikes into 100 bins of 20 ms using cluster IDs and trial-relative spike times. Multiple probes are merged by offsetting cluster IDs. The result stays as spike counts and is finally cast to `uint8`; it is not converted to firing rate in Hz.

ii.
```python
bin_idx = np.clip(((t - interval_begs[trial_idx]) / BINSIZE).astype(np.int32), 0, N_BINS - 1)
lin_idx = c * N_BINS + b
counts = np.bincount(lin_idx, minlength=minlength)
binned[trial_idx] = counts[:minlength].reshape(n_clusters_total, N_BINS)
```

```python
neural_trial = neural_data[trial_idx].astype(np.uint8)  # already (n_clusters, N_BINS)
```

iii. The notes say this “matches `bin_spiking_data()`,” but the explicit motivation given in the trajectory for the `uint8` cast was memory reduction after full-dataset runs hit the memory limit.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not filter neurons by cluster quality. All clusters in the loaded spike files are kept, and only region labels are attached afterward through Beryl mapping.

ii.
```python
n_clusters = int(merged_spikes['clusters'].max()) + 1 if len(merged_spikes['clusters']) > 0 else 0
...
if cluster_brain_ids is not None:
    cluster_acronyms = br.id2acronym(cluster_brain_ids)
    cluster_beryl = br.acronym2acronym(cluster_acronyms, mapping='Beryl')
else:
    cluster_beryl = np.array(['void'] * n_clusters)
```

```python
'neuron_filtering': 'none (all clusters used, matching Zhang et al. 2025)',
```

iii. The notes repeatedly justify this as “Follow Zhang code: use ALL neurons,” citing `qc=None` in the reference utility code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to stimulus onset. For each trial, the window start and stop are `stimOn_times + (-0.5, 1.5)`, and spike times are binned relative to those trial-specific bounds.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
stim_times = trials_df[ALIGN_TIME].values
interval_begs = stim_times + TIME_WINDOW[0]
interval_ends = stim_times + TIME_WINDOW[1]
```

```python
b = np.clip(((t - interval_begs[trial_idx]) / BINSIZE).astype(np.int32), 0, N_BINS - 1)
```

iii. The notes justify this as following the unified stimulus-onset alignment used in `0_data_caching.py` and matching the decoder instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent uses 20 ms bins across a 2 s window, giving `100` bins per trial. No additional temporal rebinning is applied.

ii.
```python
BINSIZE = 0.02
TIME_WINDOW = (-0.5, 1.5)
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The notes justify this by pointing to the reference code’s `binsize = 0.02` and the methods paper’s “20-ms bins, producing T = 100.”

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from `stimOn_times`, in the sense that each trial’s time axis is defined relative to the stimulus-onset alignment event.

ii.
```python
ALIGN_TIME = 'stimOn_times'
stim_times = trials_df[ALIGN_TIME].values
interval_begs = stim_times + TIME_WINDOW[0]
interval_ends = stim_times + TIME_WINDOW[1]
```

```python
time_since_onset = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The notes describe this as a derived alignment variable rather than a column directly stored in raw data.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent constructs one fixed 100-sample vector shared by all trials using `np.linspace(-0.48, 1.5, 100)`, then copies it into the first input row of every trial.

ii.
```python
time_since_onset = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
...
input_trial[0, :] = time_since_onset
```

iii. The notes justify this as “matching reference code: bin centers from interval_beg + binsize to interval_end,” although the implemented vector is the right-edge style grid implied by that formula.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The agent uses the same number of bins and the same trial window as the neural data, and it reuses the same fixed per-trial time vector for every trial. In the implementation, that grid runs from `-0.48` to `1.5` s, while spike counts are accumulated over bins defined from `-0.5` to `1.5` s.

ii.
```python
binned_spikes = bin_spikes_fast(
    merged_spikes['times'], merged_spikes['clusters'],
    interval_begs, interval_ends, n_clusters
)
```

```python
time_since_onset = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The notes claim this matches the reference alignment grid for all variables.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft` in the trials table. A new block is detected whenever `probabilityLeft` changes.

ii.
```python
def compute_trial_number_in_block(prob_left):
    trial_numbers = np.zeros(len(prob_left), dtype=np.float32)
    current_block_start = 0
    current_val = prob_left[0]
```

iii. The notes justify this by stating that the raw trials table has no direct block-trial counter, so it must be inferred from block-probability transitions.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The agent scans the full unfiltered `probabilityLeft` sequence, resets a block start index when the value changes, and stores `i - current_block_start` as a zero-based within-block trial number. It then applies the trial mask afterward.

ii.
```python
for i in range(len(prob_left)):
    if prob_left[i] != current_val:
        current_block_start = i
        current_val = prob_left[i]
    trial_numbers[i] = i - current_block_start
```

```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
trial_num_in_block = all_trial_nums[good_indices].astype(np.float32)
```

iii. The notes explicitly justify computing it before filtering so the counter reflects the animal’s real block position, not the filtered-trial index.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the trials table.

ii.
```python
choice_raw = trials_df['choice'].values[good_indices]       # -1 or 1
```

iii. The notes identify `trials.choice` as the source variable and treat `0` as no-choice trials that are excluded upstream.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent converts the surviving `choice` values with `((choice_raw + 1) / 2)`, which maps `-1 -> 0` and `+1 -> 1`, and then broadcasts that label across all 100 time bins of the trial.

ii.
```python
choice = ((choice_raw + 1) / 2).astype(np.int32)  # 0 or 1
...
output_trial[0, :] = choice[trial_idx]
```

iii. The notes justify this as producing the requested binary left/right encoding, but they describe the IBL sign convention incorrectly.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column of the trials table.

ii.
```python
prob_left = trials_df['probabilityLeft'].values[good_indices]
```

iii. The notes identify `probabilityLeft` as the task block prior and map it to the decoder’s categorical target.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The agent remaps the three allowed values `0.2`, `0.5`, and `0.8` to class labels `0`, `1`, and `2`, then broadcasts the per-trial label across all 100 bins.

ii.
```python
prior = np.zeros(len(prob_left), dtype=np.int32)
prior[prob_left == 0.2] = 0
prior[prob_left == 0.5] = 1
prior[prob_left == 0.8] = 2
...
output_trial[1, :] = prior[trial_idx]
```

iii. The notes justify this as directly following the task specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
pos_file = find_file(alf_path, '_ibl_wheel.position.npy')
ts_file = find_file(alf_path, '_ibl_wheel.timestamps.npy')
...
re_pos = np.load(pos_file).flatten()
re_ts = np.load(ts_file).flatten()
```

iii. The notes justify this as matching the reference’s use of wheel velocity from the IBL wheel stream.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The agent recreates wheel speed by interpolating wheel position to a uniform 1000 Hz grid, low-pass filtering with an 8th-order 20 Hz Butterworth filter, differentiating to velocity, taking the absolute value, and then linearly interpolating the result onto each trial’s 100-bin grid.

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
y_interp = interp1d(trial_times, trial_vals, kind='linear',
                   fill_value='extrapolate')(x_interp)
```

iii. The notes explicitly justify this as matching `SessionLoader.load_wheel()` and the reference behavior interpolation path.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. After masking trials, the agent flattens all wheel-speed samples from that session, computes session-specific quantile thresholds, and digitizes them into three equal-frequency categories labeled `0`, `1`, and `2`.

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

iii. The notes justify this as “3 equal-frequency (quantile) bins across all trials in a session.”

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is interpolated per trial onto `x_interp = np.linspace(t_beg + 0.02, t_end, 100)`, where `t_beg` and `t_end` are the stimulus-aligned trial bounds. This gives a 100-sample trace per kept trial using the same nominal trial window as the neural data.

ii.
```python
idxs_beg = np.searchsorted(beh_times, interval_begs, side='right')
idxs_end = np.searchsorted(beh_times, interval_ends, side='left')
...
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
y_interp = interp1d(trial_times, trial_vals, kind='linear',
                   fill_value='extrapolate')(x_interp)
```

iii. The notes justify this as matching the reference code’s behavior alignment to the stimulus-onset decoder grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from either `leftCamera.ROIMotionEnergy.npy` with `_ibl_leftCamera.times.npy`, or, if those are unavailable, `rightCamera.ROIMotionEnergy.npy` with `_ibl_rightCamera.times.npy`.

ii.
```python
me_file = find_file(alf_path, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_path, '_ibl_leftCamera.times.npy')
...
me_file = find_file(alf_path, 'rightCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_path, '_ibl_rightCamera.times.npy')
```

iii. The notes justify this as following the reference’s “left camera first, fall back to right” rule.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The agent uses the raw motion-energy trace without extra filtering, then linearly interpolates it onto each trial’s 100-sample stimulus-aligned grid.

ii.
```python
me_times, me_values = load_whisker_motion_energy(session_path)
...
binned_me, me_mask = interpolate_behavior(
    me_times, me_values, interval_begs, interval_ends)
```

```python
y_interp = interp1d(trial_times, trial_vals, kind='linear',
                   fill_value='extrapolate')(x_interp)
```

iii. The notes justify this as matching the reference code’s use of released motion-energy values plus interpolation to decoder bins.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, whisker motion energy is flattened within a session after trial filtering, split by session-specific quantiles, and digitized into three categories.

ii.
```python
me_flat = me_data.flatten()
me_discrete = discretize_to_bins(me_flat, n_bins=3)
me_discrete_2d = me_discrete.reshape(me_data.shape).astype(np.int32)
```

```python
quantiles = np.linspace(0, 100, n_bins + 1)
thresholds = np.percentile(valid, quantiles)
result = np.digitize(values, thresholds[1:-1], right=False).astype(np.int32)
```

iii. The notes justify this with the same equal-frequency binning rationale used for wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned exactly as wheel speed: it is interpolated onto the same stimulus-centered 100-sample grid from `t_beg + 0.02` to `t_end`.

ii.
```python
binned_me, me_mask = interpolate_behavior(
    me_times, me_values, interval_begs, interval_ends)
```

```python
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
y_interp = interp1d(trial_times, trial_vals, kind='linear',
                   fill_value='extrapolate')(x_interp)
```

iii. The notes justify this as the same unified decoder alignment used for all streams.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent generally drops missing or unusable data rather than repairing it. Missing trials-table files, missing spike data, missing wheel/ME streams, sessions with zero clusters, and sessions with fewer than two surviving trials all cause the session to be skipped. Missing behavior coverage becomes `False` in `wheel_mask` or `me_mask` and removes the affected trials.

ii.
```python
if trials_df is None:
    print(f"  No trials table found for {eid}")
    return None
...
if len(spikes_list) == 0:
    print(f"  No spike data found for {eid}")
    return None
...
if n_clusters == 0:
    print(f"  No clusters found for {eid}")
    return None
...
if n_good_trials < 2:
    print(f"  Too few valid trials ({n_good_trials}) for {eid}")
    return None
```

```python
if wheel_times is not None:
    binned_wheel, wheel_mask = interpolate_behavior(
        wheel_times, wheel_speed, interval_begs, interval_ends)
else:
    binned_wheel = np.full((len(trials_df), N_BINS), np.nan, dtype=np.float32)
    wheel_mask = np.zeros(len(trials_df), dtype=bool)
```

iii. The notes frame this as standard trial/session skipping for missing data, consistent with the reference workflow, though the direct-file loader also silently resolves revisions by filename search.

## 10-a. What are the most time-consuming steps of the code?

i. The agent’s own notes say spike loading is the largest cost, followed by spike binning and wheel loading. The runtime table in `CONVERSION_NOTES.md` estimates about `0.9 s` for spike loading, `0.4 s` for spike binning, and `0.5 s` for wheel loading per session.

ii.
```python
for pname in probe_names:
    spikes, clusters = load_spike_data(session_path, pname)
```

```python
binned_spikes = bin_spikes_fast(
    merged_spikes['times'], merged_spikes['clusters'],
    interval_begs, interval_ends, n_clusters
)
```

iii. The justification in the notes is empirical runtime measurement from the agent’s conversion runs.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent already partially vectorized spike binning with `np.searchsorted` and `np.bincount`, but several trial-wise and neuron-wise Python loops remain: the loop over valid trials inside `bin_spikes_fast`, the loop in `interpolate_behavior`, the loop in `compute_trial_number_in_block`, the loop that builds per-trial output arrays, and the loop that converts region names to indices.

ii.
```python
for i, trial_idx in enumerate(valid_idx):
    s, e = starts[i], ends[i]
    ...
    counts = np.bincount(lin_idx, minlength=minlength)
```

```python
for trial_idx in range(n_trials):
    ...
    input_list.append(input_trial)
    ...
    output_list.append(output_trial)
```

iii. The notes justify the chosen optimizations by saying “Code speedups added” for spike binning and vectorized `searchsorted`; there is no separate justification for leaving the remaining Python loops in place.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats some work: reaction time is recomputed twice in `create_trials_mask`, session temp files are loaded once for metadata and then again for final assembly, and discrete-threshold logic is run separately for wheel and whisker with the same helper. It also walks revision directories repeatedly through `find_file`.

ii.
```python
if MIN_RT is not None:
    rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
    mask &= (rt >= MIN_RT)
if MAX_RT is not None:
    rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
    mask &= (rt <= MAX_RT)
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

iii. There is no explicit justification for this repeated work beyond the notes’ discussion of avoiding out-of-memory failures by staging per-session temporary pickles.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads cluster depths and cluster metrics but never uses them for the converted dataset. `merge_probes` also accumulates `merged_channels` and `merged_depths` without using them. Optional plotting computes visualization summaries that are discarded after saving PNGs, and metadata such as trial-filter settings are recorded but not used by downstream decoder training.

ii.
```python
channels_file = os.path.join(spike_dir, 'clusters.channels.npy')
depths_file = os.path.join(spike_dir, 'clusters.depths.npy')
metrics_file = os.path.join(spike_dir, 'clusters.metrics.pqt')
...
if os.path.exists(depths_file):
    clusters['depths'] = np.load(depths_file).flatten()
if os.path.exists(metrics_file):
    clusters['metrics'] = pd.read_parquet(metrics_file)
```

```python
merged_channels = []
merged_depths = []
...
if clusters.get('channels') is not None:
    merged_channels.append(clusters['channels'])
if clusters.get('depths') is not None:
    merged_depths.append(clusters['depths'])
```

iii. No explicit justification was given for these unused loads; they appear to be leftovers from mimicking the broader reference environment rather than the minimum needed conversion path.
