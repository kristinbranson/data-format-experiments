# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use the ONE API as the primary loader. It read `/app/code/code_zhang2025/data/bwm_release.csv` to enumerate sessions, then manually constructed session paths under `/app/data/one_cache` and opened the raw parquet and NumPy files directly from disk. Trials, wheel, whisker motion energy, and spike-sorting outputs were each loaded by custom helper functions.

ii.
```python
DATA_ROOT = '/app/data/one_cache'
BWM_CSV = '/app/code/code_zhang2025/data/bwm_release.csv'
```

```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
session_groups = bwm_df.groupby('eid')
```

```python
session_path = os.path.join(DATA_ROOT, lab, 'Subjects', subject, date, session_num_str)
trials_df = load_trials(session_path)
wheel_times, wheel_speed = load_wheel_speed(session_path)
me_times, me_values = load_whisker_motion_energy(session_path)
```

iii. In `CONVERSION_NOTES.md` Step 6 the AI explicitly justified “Direct disk loading (bypasses ONE API since cache is read-only).”

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of `bwm_release.csv`. During assembly, the AI appends previously unseen subject names to `subject_set` in processing order and stores each session’s `subject_idx` as the index of that subject in `subject_set`.

ii.
```python
for eid, group in session_groups:
    row = group.iloc[0]
    probe_names = list(group['probe_name'])
    session_list.append((
        eid, row['lab'], row['subject'], row['date'],
        row['session_number'], probe_names
    ))
```

```python
if subject not in subject_set:
    subject_set.append(subject)
subject_idx = subject_set.index(subject)
```

iii. The notes treat the release CSV as the authoritative index of sessions and subjects. There is no extra subject parsing from file paths beyond using the CSV fields.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in `bwm_release.csv`. The AI groups the CSV by `eid`, collects all probe names for that `eid`, and processes one `(eid, subject, date, session_number, probe_names)` tuple at a time.

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

iii. The notes describe the BWM release CSV as the session index and treat one `eid` as one session, with multiple probes merged inside that session.

## 1-d. How are the data split into trials?

i. The trials are taken as the rows of `_ibl_trials.table.pqt`. The AI loads the trials parquet into a DataFrame and later indexes rows with Boolean masks and `good_indices`.

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
good_indices = np.where(combined_mask)[0]
choice_raw = trials_df['choice'].values[good_indices]
prob_left = trials_df['probabilityLeft'].values[good_indices]
```

iii. The AI’s notes describe the trials table as already containing one row per trial and use that table as the unit of trial splitting.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a composite mask. It excludes NaNs in a fixed list of trial fields, enforces reaction time between 0.08 and 2.0 s, excludes trials longer than 10 s from `goCue_times` to `feedback_times`, excludes no-choice trials, and then requires successful wheel and whisker interpolation coverage over the full aligned window.

ii.
```python
NAN_EXCLUDE = [
    'stimOn_times', 'choice', 'feedback_times',
    'probabilityLeft', 'firstMovement_times', 'feedbackType'
]
```

```python
mask = np.ones(n_trials, dtype=bool)
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
combined_mask = mask & wheel_mask & me_mask
```

iii. `CONVERSION_NOTES.md` Step 3 and Step 10 justify this as matching `load_trials_and_mask()` plus behavior interpolation masking, although the AI also added the `max_trial_len` filter and broader NaN exclusions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural array is derived from `spikes.times.npy` and `spikes.clusters.npy`, loaded probe by probe. Cluster-side arrays such as `clusters.channels.npy` and channel brain-location IDs are used for region labels, not for the spike counts themselves.

ii.
```python
times_file = os.path.join(spike_dir, 'spikes.times.npy')
clusters_file = os.path.join(spike_dir, 'spikes.clusters.npy')
spikes = {
    'times': np.load(times_file).flatten(),
    'clusters': np.load(clusters_file).flatten(),
}
```

```python
channels_file = os.path.join(spike_dir, 'clusters.channels.npy')
...
clusters['brain_ids'] = brain_ids
```

iii. The notes repeatedly describe the neural data as spike times and cluster assignments binned into decoder windows, with region labels added via Beryl mapping.

## 2-b. How is the `neural` data processed?

i. The AI merges probes within a session, sorts merged spikes by time, bins spikes into 20 ms stimulus-aligned bins, and keeps the result as integer spike counts. It does not divide by bin width to convert counts to firing rates. Before saving each trial, it casts the binned counts to `uint8` for memory savings.

ii.
```python
merged_spikes, cluster_brain_ids = merge_probes(spikes_list, clusters_list)
...
sort_idx = np.argsort(all_times, kind='stable')
merged_spikes = {
    'times': all_times[sort_idx],
    'clusters': all_clusters[sort_idx],
}
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

iii. `CONVERSION_NOTES.md` Step 6 says spike binning follows the reference pipeline and Step 9 says `uint8` was chosen to fit memory limits because “spike counts in 20ms bins are small integers 0-255.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies no neuron QC filter. It uses all clusters present in the released spike-sorting output and records this explicitly in metadata.

ii.
```python
n_clusters = int(merged_spikes['clusters'].max()) + 1 if len(merged_spikes['clusters']) > 0 else 0
```

```python
'neuron_filtering': 'none (all clusters used, matching Zhang et al. 2025)',
```

iii. `CONVERSION_NOTES.md` Step 4 and Step 10 explicitly justify this as following the Zhang code’s `qc=None`, even though the notes also acknowledge the data paper’s well-isolated-neuron QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`. For each trial, the AI defines `[stimOn_times - 0.5, stimOn_times + 1.5]` as the binning window and bins spikes relative to that window.

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

iii. The notes justify a unified stimulus-onset alignment based on the task instructions and on the AI’s reading of `0_data_caching.py`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 20 ms bins across a 2 s window, giving 100 bins per trial. There is no coarser rebinning or smoothing.

ii.
```python
BINSIZE = 0.02
TIME_WINDOW = (-0.5, 1.5)
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. `CONVERSION_NOTES.md` Step 3 and Step 10 justify this as matching the reference code’s unified 20 ms binning.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the alignment variable `stimOn_times` plus the fixed decoding window and bin size. The stored input is not copied from a raw time-series file; it is constructed from the aligned trial geometry.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
```

```python
stim_times = trials_df[ALIGN_TIME].values
interval_begs = stim_times + TIME_WINDOW[0]
interval_ends = stim_times + TIME_WINDOW[1]
```

iii. The notes say all variables are aligned to `stimOn_times`, so the time input is defined from that alignment rather than loaded as an independent raw signal.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI creates a fixed 100-point vector from `-0.48` to `1.5` seconds using `np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)` and reuses that same vector for every trial.

ii.
```python
time_since_onset = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The comment in code says this is “Matching reference code: bin centers from interval_beg + binsize to interval_end,” and the notes frame it as the common decoder time axis.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The AI intends the time input to share the neural trial window and behavior interpolation grid. In practice it uses the same `[t_beg, t_end]` window as the neural data, but labels time with points from `t_beg + 0.02` through `t_end`, i.e. effectively the right edges of bins rather than bin centers.

ii.
```python
interval_begs = stim_times + TIME_WINDOW[0]
interval_ends = stim_times + TIME_WINDOW[1]
```

```python
time_since_onset = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The AI justified a shared time axis for neural and behavior streams, but its implementation uses the same grid as its behavior interpolation rather than the reference bin-center grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft`, with a new block starting whenever `probabilityLeft` changes.

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
    return trial_numbers
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly states that trial number in block is computed from transitions in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes the count on the full unfiltered `probabilityLeft` sequence, so excluded trials still advance the within-block counter. It then indexes this per-trial count by the surviving `good_indices` and broadcasts the scalar across time bins.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
trial_num_in_block = all_trial_nums[good_indices].astype(np.float32)
```

```python
input_trial[1, :] = trial_num_in_block[trial_idx]
```

iii. The notes explicitly say “use full trials_df for correct block computation,” which is the same rationale the human reference gives.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the `choice` column of the trials table, after trial filtering. The AI treats `-1` as left and `1` as right.

ii.
```python
choice_raw = trials_df['choice'].values[good_indices]
choice = ((choice_raw + 1) / 2).astype(np.int32)
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly states “Choice mapping: IBL choice -1 (left) → 0, choice 1 (right) → 1.”

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI recodes the surviving `choice` values with `(-1 -> 0, 1 -> 1)` and then broadcasts the result across all 100 time bins of each trial.

ii.
```python
choice = ((choice_raw + 1) / 2).astype(np.int32)
```

```python
output_trial[0, :] = choice[trial_idx]
```

iii. The notes justify this as the binary choice representation required by the decoder task.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column of the trials table.

ii.
```python
prob_left = trials_df['probabilityLeft'].values[good_indices]
```

iii. The notes identify task “prior” with the trials-table block prior, i.e. `probabilityLeft`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then broadcasts that per-trial label across all time bins.

ii.
```python
prior = np.zeros(len(prob_left), dtype=np.int32)
prior[prob_left == 0.2] = 0
prior[prob_left == 0.5] = 1
prior[prob_left == 0.8] = 2
```

```python
output_trial[1, :] = prior[trial_idx]
```

iii. `CONVERSION_NOTES.md` Step 5 says this mapping is chosen to match the decoder specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`. The AI reconstructs velocity from position and timestamps, then takes absolute velocity as speed.

ii.
```python
pos_file = find_file(alf_path, '_ibl_wheel.position.npy')
ts_file = find_file(alf_path, '_ibl_wheel.timestamps.npy')
```

```python
vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, position)), 0, 0) * fs
speed = np.abs(vel).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 6 explicitly justifies this as matching brainbox wheel processing and `wheel-speed = abs(velocity)`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI interpolates wheel position onto a 1000 Hz uniform grid, applies an order-8 20 Hz Butterworth low-pass filter, differentiates to get velocity, takes absolute velocity, and linearly interpolates the resulting speed trace onto each trial’s aligned 100-bin grid.

ii.
```python
fs = 1000
t = np.arange(re_ts[0], re_ts[-1], 1.0 / fs)
position = scipy_interp1d(re_ts, re_pos, kind='linear')(t)
```

```python
corner_frequency = 20
order = 8
sos = scipy.signal.butter(N=order, Wn=corner_frequency / fs * 2,
                           btype='lowpass', output='sos')
vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, position)), 0, 0) * fs
speed = np.abs(vel).astype(np.float32)
```

```python
y_interp = interp1d(trial_times, trial_vals, kind='linear',
                    fill_value='extrapolate')(x_interp)
```

iii. The notes describe this as intentionally matching `SessionLoader.load_wheel()` and the reference behavior interpolation logic.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. After masking to valid trials, the AI flattens all wheel-speed samples within a session, computes session-specific quantile thresholds, digitizes into three equal-frequency bins, and reshapes back to `(n_trials, N_BINS)`.

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

iii. `CONVERSION_NOTES.md` Step 5 and Step 7 explicitly justify this as 3 equal-frequency bins per session.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The AI aligns wheel speed to the same stimulus-centered trial window as the neural data and interpolates it onto the same 100 stored time points used for the input time axis. Those stored points run from `t_beg + 0.02` to `t_end`.

ii.
```python
combined_mask = mask & wheel_mask & me_mask
```

```python
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
y_interp = interp1d(trial_times, trial_vals, kind='linear',
                    fill_value='extrapolate')(x_interp)
```

iii. The AI’s notes justify unified stimulus-onset alignment for all variables, and the code uses the same interpolation grid for both the time input and wheel output.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` plus `_ibl_leftCamera.times.npy` when available, otherwise from the right-camera equivalents.

ii.
```python
me_file = find_file(alf_path, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_path, '_ibl_leftCamera.times.npy')
...
me_file = find_file(alf_path, 'rightCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_path, '_ibl_rightCamera.times.npy')
```

iii. `CONVERSION_NOTES.md` Step 5 and Step 6 explicitly state “Try left camera first, fall back to right.”

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI loads the released motion-energy trace without additional filtering, then linearly interpolates it onto each trial’s 100 aligned time points before discretization.

ii.
```python
if me_times is not None:
    binned_me, me_mask = interpolate_behavior(
        me_times, me_values, interval_begs, interval_ends)
```

```python
y_interp = interp1d(trial_times, trial_vals, kind='linear',
                    fill_value='extrapolate')(x_interp)
```

iii. The notes say whisker motion energy should follow the reference code’s left-preferred loading and standard continuous-trace interpolation, with no extra normalization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is thresholded the same way as wheel speed: flatten the session’s valid samples, compute session-level quantile thresholds, digitize to three categories, and reshape back to trial-by-time form.

ii.
```python
me_flat = me_data.flatten()
me_discrete = discretize_to_bins(me_flat, n_bins=3)
me_discrete_2d = me_discrete.reshape(me_data.shape).astype(np.int32)
```

iii. The notes explicitly justify using the same equal-frequency 3-bin discretization for whisker motion energy as for wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The AI aligns whisker motion energy to the same stimulus-centered trial window as the neural data and interpolates it onto the same stored 100-point grid from `t_beg + 0.02` to `t_end`.

ii.
```python
if me_times is not None:
    binned_me, me_mask = interpolate_behavior(
        me_times, me_values, interval_begs, interval_ends)
```

```python
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
```

iii. The notes justify a unified stimulus-onset alignment for all decoder variables.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing files usually cause a probe or whole session to be skipped. Missing or mismatched behavior traces yield all-false masks so the session can fall below the 2-trial minimum and be dropped. Missing trial fields are handled by NaN-based exclusion. For revisioned files, the AI searches the base directory and then revision directories in descending order.

ii.
```python
def find_file(base_path, filename, revisions=None):
    direct = os.path.join(base_path, filename)
    if os.path.exists(direct):
        return direct
    ...
    for rev in revisions:
        candidate = os.path.join(base_path, rev, filename)
        if os.path.exists(candidate):
            return candidate
```

```python
if pos_file is None or ts_file is None:
    return None, None
...
if me_file is not None and times_file is not None:
    ...
return None, None
```

```python
if n_good_trials < 2:
    print(f"  Too few valid trials ({n_good_trials}) for {eid}")
    return None
```

iii. `CONVERSION_NOTES.md` Step 9 explains that skipped sessions were overwhelmingly due to ending up with zero valid trials after trial masking plus behavior coverage checks.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identified loading spike files as the main cost, with wheel loading and spike binning next. It instrumented per-session timing prints for spike binning, behavior processing, and total session time.

ii.
```python
t_spike = time.time()
binned_spikes = bin_spikes_fast(...)
t_spike_done = time.time()
...
t_beh_done = time.time()
...
print(f"  {eid}: {n_trials} trials, {n_clusters} neurons | "
      f"spike_bin={t_spike_done-t_spike:.1f}s, beh={t_beh_done-t_spike_done:.1f}s, "
      f"total={t_format_done-t0:.1f}s")
```

iii. `CONVERSION_NOTES.md` Step 7 reports runtime estimates of roughly 0.9 s/session to load spikes, 0.5 s for wheel loading, and 0.4 s for spike binning.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI partially vectorized spike binning with `np.bincount`, but it still leaves several per-trial Python loops: trial-by-trial interpolation in `interpolate_behavior`, trial-by-trial formatting of `neural`/`input`/`output`, and the per-neuron loop that builds `brain_region_idx`.

ii.
```python
for trial_idx in range(n_trials):
    ...
    input_list.append(input_trial)
    output_list.append(output_trial)
```

```python
for trial_idx in range(n_trials):
    ...
    y_interp = interp1d(trial_times, trial_vals, kind='linear',
                        fill_value='extrapolate')(x_interp)
```

```python
for neuron_idx in range(result['n_neurons']):
    region = result['cluster_beryl'][neuron_idx]
    ...
```

iii. The notes emphasize the speedup from replacing slower spike accumulation with `np.bincount`, but the remaining loops were left in straightforward Python.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several pieces of work: reaction time is recomputed twice in `create_trials_mask`, revision-directory searching is repeated for many files through `find_file`, and the conversion writes every session to a temp pickle and then reads those temp files back twice during reassembly.

ii.
```python
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
```

```python
for batch_start in range(0, len(session_meta), BATCH_SIZE):
    ...
    with open(session_meta[idx]['tmp_file'], 'rb') as f:
        sess = pickle.load(f)
```

iii. `CONVERSION_NOTES.md` Step 9 explains that temp-file roundtripping was introduced as a memory workaround after earlier in-memory assembly runs were killed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The biggest added work is temp-file staging and batch reloading solely to survive memory limits. The script also supports optional diagnostic plotting and stores large metadata structures that are not used by the downstream decoder. Finally, it converts neural data to `uint8`, which the verifier later warns will be converted back during training.

ii.
```python
tmp_file = os.path.join(tmp_dir, f'session_{n_processed:04d}.pkl')
with open(tmp_file, 'wb') as f:
    pickle.dump({...}, f, protocol=pickle.HIGHEST_PROTOCOL)
```

```python
if show_processing:
    plot_processing(...)
```

```python
neural_trial = neural_data[trial_idx].astype(np.uint8)
```

iii. The trajectory and notes show this extra work was added to work around 64 GB memory failures. The later verification warning explicitly noted that `uint8` neural data would be converted during training.
