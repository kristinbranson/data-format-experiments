# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not use the ONE API. It loads a session list from `bwm_release.csv`, resolves cache paths from `lab/subject/date`, reads parquet and `.npy` files directly from `/app/data/one_cache`, and processes each session by walking probe directories and behavior files under `alf/`.

ii. ```python
DATA_DIR = Path('/app/data/one_cache')
BWM_CSV = Path('/app/code/code_zhang2025/data/bwm_release.csv')
```

```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
sessions = get_session_list(bwm_df)
```

```python
alf_path = find_session_path(lab, subject, date)
trials = load_trials(alf_path)
wheel_pos_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.position.npy')).flatten()
wheel_ts_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.timestamps.npy')).flatten()
```

iii. In `CONVERSION_NOTES.md`, the agent justified this as “Direct file loading (no ONE API needed, works offline from cache)” and treated the BWM release CSV as the ground-truth session list.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the session rows in the BWM release CSV and from the per-session `subject` field in each processed result. Final `subjects` is a sorted unique list, and `subject_idx` maps each session to that subject list.

ii. ```python
eid = session_info['eid']
subject = session_info['subject']
lab = session_info['lab']
date = session_info['date']
```

```python
all_subjects = sorted(set(r['subject'] for r in all_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
subject_idx.append(subject_to_idx[r['subject']])
```

iii. The notes say “Session list: Use BWM release CSV (459 sessions) as ground truth,” so the justification is that subject identity is already present in that release metadata.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values from `bwm_release.csv`. The agent groups the release table by `eid`, takes one row per session, and processes sessions one at a time.

ii. ```python
def get_session_list(bwm_df):
    sessions = bwm_df.groupby('eid').first().reset_index()
    return sessions
```

```python
for idx in range(len(sessions)):
    sess = sessions.iloc[idx]
    result = process_session(sess, br, show_processing=args.show_processing)
```

iii. The justification in the notes is that the BWM release CSV resolves the paper/code discrepancy between 433 and 459 sessions, so it is used as the session index.

## 1-d. How are the data split into trials?

i. Trials are taken from the rows of the session trial table `_ibl_trials.table.pqt`. After loading the table, the agent filters rows with a boolean mask and then uses the remaining rows as the trial list for the session.

ii. ```python
def load_trials(alf_path):
    trials_file = find_latest_revision(alf_path, '_ibl_trials.table.pqt')
    trials = pd.read_parquet(trials_file)
    return trials
```

```python
mask = create_trial_mask(trials)
valid_trials = trials[mask].reset_index(drop=True)
```

iii. The agent’s notes describe the trial table as the source of “Trial events (stimOn, choice, feedback, etc.),” and the code follows that directly.

## 1-e. How are trials filtered based on quality controls?

i. The agent filters trials by removing NaNs in several trial columns, keeping reaction times in `[0.08, 2.0]`, excluding `choice == 0`, excluding trials with `feedback_times - goCue_times > 10 s`, and then removing any remaining trial whose wheel or motion-energy trace cannot be interpolated without NaNs over the full aligned window.

ii. ```python
for event in NAN_EXCLUDE:
    if event in trials.columns:
        mask &= ~trials[event].isna()

rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
mask &= (trials['choice'] != 0)
```

```python
if 'goCue_times' in trials.columns:
    trial_len = trials['feedback_times'] - trials['goCue_times']
    mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()
```

```python
combined_mask = np.ones(len(valid_trials), dtype=bool)
combined_mask &= ~np.any(np.isnan(wheel_speed_binned), axis=1)
combined_mask &= ~np.any(np.isnan(me_binned), axis=1)
```

iii. The notes say the trial mask “matches reference code exactly,” and also state that wheel and whisker trials are dropped unless all aligned data are available.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are derived from probe-level `spikes.times.npy` and `spikes.clusters.npy`. The code also loads cluster/channel metadata to count clusters and assign brain regions.

ii. ```python
spikes = {
    'times': np.load(rev_dir / 'spikes.times.npy').flatten(),
    'clusters': np.load(rev_dir / 'spikes.clusters.npy').flatten(),
}
```

```python
chan_brain_ids = np.load(rev_dir / 'channels.brainLocationIds_ccf_2017.npy').flatten()
clusters = {
    'channels': clusters_channels,
    'depths': clusters_depths,
    'metrics': metrics,
    'chan_brain_ids': chan_brain_ids,
}
```

iii. The notes map `spikes.times + spikes.clusters -> neural`, with brain-region information added separately through Beryl mapping.

## 2-b. How is the `neural` data processed?

i. The agent merges probes by offsetting cluster IDs, sorts all spikes by time, bins spikes into 20 ms bins over a `(-0.5, 1.5)` stimulus-aligned window, and stores the per-trial `(n_clusters, 100)` arrays as `uint8` spike counts. It does not divide by bin width to convert counts to firing rates.

ii. ```python
merged_spikes_clusters.append(spikes['clusters'] + cluster_offset)
sort_idx = np.argsort(all_times, kind='stable')
```

```python
bin_idx = np.minimum(((t_sel - t_start) / binsize).astype(np.int32), n_bins - 1)
np.add.at(binned[trial_idx].ravel(), flat_idx, 1)
```

```python
for t in range(len(final_spikes)):
    neural_trials.append(final_spikes[t].astype(np.uint8))
```

iii. The notes justify 20 ms bins and session-wide probe merging by reference-code consistency, and justify `uint8` storage implicitly as a memory-saving choice.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent keeps all clusters that appear in the spike-sorting output. It does not apply a cluster-quality threshold such as `label >= 1`, and it does not exclude `void` regions from the neural matrix.

ii. ```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
n_clusters = len(clusters['channels'])
cluster_regions = get_brain_regions(clusters, br)
```

There is no code that filters `clusters['metrics']`, `label`, or `void`.

iii. The notes explicitly justify this as “Use all clusters (qc=None): Matches reference code, not just well-isolated neurons.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`. For every kept trial the code uses `stimOn_times + (-0.5, 1.5)` as the window and bins spikes relative to that interval.

ii. ```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

```python
align_times = valid_trials[ALIGN_TIME].values
interval_starts = align_times + TIME_WINDOW[0]
interval_ends = align_times + TIME_WINDOW[1]
```

iii. The notes state “Align everything to stimOn_times,” citing the task spec and the reference code’s stimulus-onset alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins over a 2 s window, for 100 bins per trial. There is no further temporal rebinning after the initial spike binning / behavioral interpolation.

ii. ```python
BINSIZE = 0.02
TIME_WINDOW = (-0.5, 1.5)
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. The notes justify this with “Time window (-0.5, 1.5)” and “20ms for all,” matching the reference code’s main settings.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The time-since-stimulus input is not read from a raw data column; it is constructed from the chosen alignment window and bin size. `stimOn_times` is used only to place the trial windows, not to compute varying values per bin.

ii. ```python
ALIGN_TIME = 'stimOn_times'
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. The notes justify this as a manually constructed decoder input: “Time since stimOn ... same for all trials.”

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code creates a fixed 100-sample vector with `np.linspace(-0.48, 1.5, 100)` and reuses it for every trial. This corresponds to right-edge samples of the 20 ms bins rather than bin centers.

ii. ```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

```python
inp = np.array([
    time_since_stim,
    np.full(N_BINS, trial_in_block_final[t], dtype=np.float32),
], dtype=np.float32)
```

iii. The notes describe this simply as “np.linspace(-0.5, 1.48, 100), continuous time-varying,” with no additional justification beyond matching the intended stimulus-aligned grid.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The agent intends it to share the same stimulus-aligned 100-bin trial window as the neural data, but in code it uses a fixed vector from `t_start + binsize` to `t_end`, not neural bin centers.

ii. ```python
interval_starts = align_times + TIME_WINDOW[0]
interval_ends = align_times + TIME_WINDOW[1]
```

```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. The notes justify alignment at a high level with “ALL alignment to stimulus onset,” but do not discuss the center-versus-edge distinction.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial-table `probabilityLeft` sequence. A change in `probabilityLeft` marks a new block.

ii. ```python
def compute_trial_in_block(prob_left):
    """Compute trial number within each block.
    Block boundaries are where probabilityLeft changes."""
```

```python
prob_left_all = trials['probabilityLeft'].values
trial_in_block_all = compute_trial_in_block(prob_left_all)
```

iii. The notes explicitly say “Trial number in block: Computed by detecting block boundaries from changes in probabilityLeft.”

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code scans the full session’s `probabilityLeft` vector, resets the counter when the value changes, starts counting at `1`, and then applies the trial-validity masks afterward. The value is broadcast across all 100 bins of the kept trial.

ii. ```python
count = 1
for i in range(len(prob_left)):
    if i > 0 and prob_left[i] != prob_left[i-1]:
        count = 1
    trial_in_block[i] = count
    count += 1
```

```python
trial_in_block_valid = trial_in_block_all[mask.values]
trial_in_block_final = trial_in_block_valid[combined_mask]
```

iii. The notes justify the block counter as “Count trials within each block” and emphasize that it is computed from the full session, then carried through masking.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the trial-table `choice` column.

ii. ```python
choice = final_trials['choice'].values.copy()
```

iii. The notes list `trials.choice -> output[0]`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code assumes the IBL convention is `-1 = left, 1 = right`, and maps it to decoder labels with `((choice + 1) // 2)`, giving `-1 -> 0` and `1 -> 1`. It excludes `choice == 0` earlier in the trial mask.

ii. ```python
mask &= (trials['choice'] != 0)
```

```python
choice_encoded = ((choice + 1) // 2).astype(int)  # -1->0, 1->1
```

iii. The justification in the notes is “Choice: left=0, right=1,” but the code comment shows the agent’s specific sign convention assumption.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trial-table `probabilityLeft` column.

ii. ```python
prob_left = final_trials['probabilityLeft'].values
```

iii. The notes list `trials.probabilityLeft -> output[1]`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code recodes `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then broadcasts that class label across the trial’s 100 time bins.

ii. ```python
prior_encoded = np.zeros(len(prob_left), dtype=int)
prior_encoded[prob_left == 0.2] = 0
prior_encoded[prob_left == 0.5] = 1
prior_encoded[prob_left == 0.8] = 2
```

```python
np.full(N_BINS, prior_encoded[t], dtype=int)
```

iii. The notes justify this directly from the task spec: “0.2->0, 0.5->1, 0.8->2.”

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. ```python
wheel_pos_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.position.npy')).flatten()
wheel_ts_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.timestamps.npy')).flatten()
wheel_times, wheel_speed = interpolate_wheel(wheel_ts_raw, wheel_pos_raw)
```

iii. The notes describe this as “wheel velocity via Butterworth filter matching ibllib.”

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code linearly interpolates wheel position to a 1 kHz regular grid, applies an 8th-order 20 Hz Butterworth low-pass filter, differentiates to obtain velocity, takes absolute value to get speed, and linearly interpolates that speed onto the trial bins.

ii. ```python
t = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
pos_interp = interp1d(timestamps, position, kind='linear')(t)
sos = signal.butter(N=order, Wn=corner_freq / fs * 2, btype='lowpass', output='sos')
vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
return t, np.abs(vel).astype(np.float32)
```

```python
wheel_speed_binned, wheel_good = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

iii. The justification in the notes is that this matches `ibllib`/reference behavior loading for wheel velocity.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The code flattens all wheel-speed values from the kept trials in a session, computes session-level quantile boundaries for 3 equal-frequency bins, digitizes the values, then reshapes back to trial-by-time arrays.

ii. ```python
all_wheel_flat = final_wheel.flatten()
wheel_disc, wheel_boundaries = discretize_to_bins(all_wheel_flat, N_DISCRETE_BINS)
wheel_disc_2d = wheel_disc.reshape(final_wheel.shape).astype(int)
```

```python
quantiles = np.linspace(0, 100, n_bins + 1)
boundaries = np.percentile(valid, quantiles)
result = np.digitize(values, boundaries[1:-1])
```

iii. The notes justify this as “equal-frequency (quantile) bins across all trials within a session, 3 bins (low/medium/high).”

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same stimulus-onset trial windows used for the neural data, then interpolated to 100 uniformly spaced samples from `t_start + binsize` to `t_end`.

ii. ```python
interval_starts = align_times + TIME_WINDOW[0]
interval_ends = align_times + TIME_WINDOW[1]
```

```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
y_interp = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. The notes justify this with the task-level decision to “Align everything to stimOn_times.”

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` plus `_ibl_leftCamera.times.npy`, with fallback to the right camera if the left camera is missing.

ii. ```python
if side == 'left':
    me_file = find_latest_revision(alf_path, 'leftCamera.ROIMotionEnergy.npy')
    times_file = find_latest_revision(alf_path, '_ibl_leftCamera.times.npy')
else:
    me_file = find_latest_revision(alf_path, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_latest_revision(alf_path, '_ibl_rightCamera.times.npy')
```

```python
me_times, me_values = load_motion_energy(alf_path, side='left')
if me_times is None:
    me_times, me_values = load_motion_energy(alf_path, side='right')
```

iii. The notes justify this as “Motion energy loading from leftCamera (fallback to rightCamera).”

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The code loads the released motion-energy trace, truncates the data to the shorter of the value/time arrays if lengths mismatch, then interpolates the trace onto the 100 trial bins. No additional filtering or normalization is applied before discretization.

ii. ```python
me = np.load(me_file).flatten()
times = np.load(times_file).flatten()
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len].astype(np.float32)
```

```python
me_binned, me_good = interpolate_behavior_to_bins(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

iii. The notes justify this as using whisker motion energy directly from the camera signal, then discretizing it per the task spec.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, the code uses session-level quantile thresholds over all kept whisker-motion-energy time points, digitizing them into 3 equal-frequency bins.

ii. ```python
all_me_flat = final_me.flatten()
me_disc, me_boundaries = discretize_to_bins(all_me_flat, N_DISCRETE_BINS)
me_disc_2d = me_disc.reshape(final_me.shape).astype(int)
```

```python
boundaries = np.percentile(valid, quantiles)
result = np.digitize(values, boundaries[1:-1])
```

iii. The notes justify this as “3 quantile bins” shared across the session.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned to the same stimulus-onset trial windows as the neural data, then sampled on the same 100 uniformly spaced interpolation points used for wheel speed.

ii. ```python
interval_starts = align_times + TIME_WINDOW[0]
interval_ends = align_times + TIME_WINDOW[1]
```

```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
y_interp = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. The notes justify this with the same global alignment decision: “ALL alignment to stimulus onset.”

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed data are mostly handled by dropping them: sessions without a path, trials file, probes, or enough valid trials are skipped; wheel/whisker trials that cannot be fully interpolated are discarded; and motion-energy length mismatches are handled by truncating both arrays to their common minimum length.

ii. ```python
if alf_path is None:
    return None
...
if not probe_dirs:
    return None
...
if len(valid_trials) < 2:
    return None
...
if n_valid < 2:
    return None
```

```python
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len].astype(np.float32)
```

iii. The notes describe this as ensuring “trials where all data is available,” and the code’s error handling shows that the general policy is skip or truncate rather than repair.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are session-by-session disk I/O for spike sorting and behavior files, the per-trial spike binning loop, and the wheel interpolation/filtering step. The code also processes sessions sequentially despite importing parallel-execution utilities.

ii. ```python
for probe_name in probe_dirs:
    spk, clu = load_spike_sorting(alf_path, probe_name)
```

```python
for trial_idx in range(n_trials):
    idx_beg = np.searchsorted(spike_times, t_start, side='left')
    ...
    np.add.at(binned[trial_idx].ravel(), flat_idx, 1)
```

```python
wheel_times, wheel_speed = interpolate_wheel(wheel_ts_raw, wheel_pos_raw)
for idx in range(len(sessions)):
    result = process_session(sess, br, show_processing=args.show_processing)
```

iii. The notes estimate about 20 s per session and emphasize direct spike loading, vectorized binning, and wheel processing as the heavy parts.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization targets are the per-trial spike binning loop, the per-trial behavior interpolation loop, the `compute_trial_in_block` loop, and the loops that build lists of per-trial `neural`, `input`, and `output` arrays.

ii. ```python
for trial_idx in range(n_trials):
    ...
    np.add.at(binned[trial_idx].ravel(), flat_idx, 1)
```

```python
for trial_idx in range(n_trials):
    ...
    y_interp = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')(x_interp)
```

```python
for i in range(len(prob_left)):
    ...
for t in range(len(final_trials)):
    ...
```

iii. The code comments call the spike binning “Optimized implementation,” so the agent appears to have considered efficiency, but several trial-wise Python loops remain.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly scans directories to find the “latest revision” files, repeats near-identical interpolation logic for wheel and motion energy, and makes multiple passes over trial masks and per-trial list assembly. It also resolves left-camera files first and may then repeat the same loading logic for the right camera.

ii. ```python
wheel_pos_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.position.npy')).flatten()
wheel_ts_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.timestamps.npy')).flatten()
```

```python
me_times, me_values = load_motion_energy(alf_path, side='left')
if me_times is None:
    me_times, me_values = load_motion_energy(alf_path, side='right')
```

```python
trial_in_block_valid = trial_in_block_all[mask.values]
trial_in_block_final = trial_in_block_valid[combined_mask]
```

iii. There is no explicit justification in the notes for this repetition; it follows from the direct-file-loading design and the session-by-session procedural structure.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads cluster depths and metrics although downstream assembly only uses cluster count and region labels; it returns quantile boundaries from discretization and `wheel_good` / `me_good` masks but never uses them; and it includes optional plotting code unrelated to the saved dataset.

ii. ```python
clusters_depths = np.load(rev_dir / 'clusters.depths.npy').flatten()
metrics_file = rev_dir / 'clusters.metrics.pqt'
...
'depths': clusters_depths,
'metrics': metrics,
```

```python
wheel_speed_binned, wheel_good = interpolate_behavior_to_bins(...)
me_binned, me_good = interpolate_behavior_to_bins(...)
wheel_disc, wheel_boundaries = discretize_to_bins(...)
me_disc, me_boundaries = discretize_to_bins(...)
```

```python
if show_processing:
    plot_processing(...)
```

iii. The notes do not justify these extras. They are side effects of implementing a more general loader / debugging workflow than the final pickle strictly needs.
