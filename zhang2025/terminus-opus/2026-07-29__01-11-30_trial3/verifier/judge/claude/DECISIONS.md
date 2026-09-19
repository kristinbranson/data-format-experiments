# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading files directly from the ONE cache directory structure on disk, rather than using the ONE API. It reads a CSV file (`bwm_release.csv`) from the reference code directory to get session metadata (eid, subject, date, lab, probe names), then builds a mapping from (subject, date) to session directory paths by walking the filesystem (`data/one_cache/<lab>/Subjects/<subject>/<date>/<session_num>/alf/`).

ii.
```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
session_map = build_session_map(DATA_DIR)

for eid, group in bwm_df.groupby('eid'):
    subject = group['subject'].iloc[0]
    date = group['date'].iloc[0]
    probe_names = list(group['probe_name'].unique())
    key = (subject, date)
    if key in session_map:
        sessions_info.append({...})
```

iii. The AI chose to load data directly from files rather than using the ONE API, which avoids needing network authentication or API setup. The session list is driven by `bwm_release.csv`.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column of `bwm_release.csv`. A running dictionary maps subject names to indices as sessions are processed in order.

ii.
```python
subject = group['subject'].iloc[0]
...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(all_subjects)
    all_subjects.append(subject)
```

iii. Subjects are assigned indices in the order they are first encountered during processing, not sorted alphabetically.

## 1-c. How are the data split into sessions?

i. Sessions are identified by grouping the `bwm_release.csv` by `eid`. Each unique `eid` becomes one session if a matching directory is found in the cache.

ii.
```python
for eid, group in bwm_df.groupby('eid'):
    ...
    if key in session_map:
        sessions_info.append({...})
```

iii. The `bwm_release.csv` already lists sessions individually.

## 1-d. How are the data split into trials?

i. Trials are loaded from the `_ibl_trials.table.pqt` parquet file, which has one row per trial.

ii.
```python
trials_df = pd.read_parquet(trials_file)
```

iii. The trials table is already organized one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using a combined mask that excludes: (1) reaction times outside 80 ms to 2 s, (2) trial durations > 10 s (`feedback_times - goCue_times > 10`), (3) trials with NaN in stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, or feedbackType, (4) no-choice trials (choice == 0), and (5) trials where wheel or whisker ME interpolation fails (via `combined_valid` mask). Unbiased block trials (probabilityLeft == 0.5) are kept.

ii.
```python
def create_trials_mask(trials_df):
    query_parts = []
    if MIN_RT is not None:
        query_parts.append(f'(firstMovement_times - stimOn_times < {MIN_RT})')
    if MAX_RT is not None:
        query_parts.append(f'(firstMovement_times - stimOn_times > {MAX_RT})')
    if MAX_TRIAL_LEN is not None:
        query_parts.append(f'(feedback_times - goCue_times > {MAX_TRIAL_LEN})')
    for event in NAN_EXCLUDE:
        query_parts.append(f'{event}.isnull()')
    if EXCLUDE_NOCHOICE:
        query_parts.append('(choice == 0)')
    query = ' | '.join(query_parts)
    mask = ~trials_df.eval(query)
    return mask
```

iii. The AI followed the reference code's `load_trials_and_mask` defaults for trial filtering. The NaN exclusion list and max_trial_len come from the reference code. The AI justified including unbiased trials because the decoder task specifies 0.5 as a valid prior class. A secondary filter drops trials where behavioral interpolation fails (`combined_valid`), serving a similar role to the reference's `covered()` check.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Spike times (`spikes.times.npy`) and spike cluster IDs (`spikes.clusters.npy`) from each probe's pykilosort output directory. Cluster-to-channel mapping (`clusters.channels.npy`) and channel brain region IDs (`channels.brainLocationIds_ccf_2017.npy`) are used for region assignment.

ii.
```python
spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(version_dir, 'spikes.clusters.npy')).flatten()
cluster_channels = np.load(os.path.join(version_dir, 'clusters.channels.npy')).flatten()
channel_brain_ids = np.load(os.path.join(version_dir, 'channels.brainLocationIds_ccf_2017.npy')).flatten()
```

iii. These are the standard spike sorting output files.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20 ms bins over the 2 s trial window (-0.5 to 1.5 s around stimulus onset), giving spike counts per cluster per bin. The counts are stored as float32. When a session has multiple probes, clusters are merged with an offset to keep cluster IDs unique. **No division by bin width** is applied — the neural data is spike counts, not firing rates.

ii.
```python
binned_spikes = bin_spikes_fast(
    spike_times, spike_clusters, n_clusters,
    interval_begs, interval_ends, BINSIZE, N_BINS
)
...
neural_list.append(final_spikes[t].astype(np.float32))
```

iii. The AI's CONVERSION_NOTES state the data is binned into 20 ms bins matching the reference. The AI did not convert counts to rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No quality control filtering is applied.** All clusters from the spike sorting are used, regardless of quality label. The AI explicitly decided on this based on the reference code calling `load_spiking_data` without a `qc` argument.

ii.
```python
# In load_spikes: all clusters are loaded
spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(version_dir, 'spikes.clusters.npy')).flatten()
# No filtering on cluster quality
```

iii. The AI's CONVERSION_NOTES state: "No QC filtering: `load_spiking_data` called without qc parameter (defaults to None = all clusters)". The AI also does not filter out `void` regions (channels outside the brain).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's time window is computed as `stimOn_times + TIME_WINDOW[0]` to `stimOn_times + TIME_WINDOW[1]`, i.e. -0.5 s to 1.5 s around stimulus onset. Spikes within this absolute time window are selected and binned.

ii.
```python
stim_on = valid_trials_df[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
...
# In bin_spikes_fast:
i_start = np.searchsorted(spike_times, t_beg, side='left')
i_end = np.searchsorted(spike_times, t_end, side='left')
bin_idx = np.minimum(((trial_times - t_beg) / binsize).astype(np.int32), n_bins - 1)
```

iii. The AI aligns to stimulus onset as specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20 ms, giving 100 bins over the 2 s window. No rebinning is applied.

ii.
```python
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 time bins
```

iii. This matches the reference code and papers.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from raw data variables. It is computed as the centers of the 100 time bins spanning the trial window.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
).astype(np.float32)
```

iii. This is a synthetic variable defined by the binning grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time values are computed as `np.linspace(-0.49, 1.49, 100)`, which gives the bin centers from -0.49 to 1.49 in 100 evenly spaced steps.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
).astype(np.float32)
```

iii. No processing beyond computing the bin centers.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input represents the center of each neural bin, so they are aligned by definition.

ii.
```python
# Same N_BINS = 100 for both neural and time input
inp = np.stack([time_since_stim, ...], axis=0)
```

iii. The time values serve as the shared time axis.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table, detecting block boundaries where the value changes.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
```

iii. The trials table carries no explicit block identifier, so blocks are inferred from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial number is the position within its block, counting from 1 (not 0). It is computed from all trials before filtering, so filtered-out trials still advance the count.

ii.
```python
def compute_trial_number_in_block(prob_left):
    trial_nums = np.zeros(len(prob_left), dtype=np.float32)
    current_block_start = 0
    current_prob = prob_left[0]
    for i in range(len(prob_left)):
        if prob_left[i] != current_prob:
            current_block_start = i
            current_prob = prob_left[i]
        trial_nums[i] = i - current_block_start + 1
    return trial_nums
```

iii. The AI computes block numbering from ALL trials before filtering, so the trial number reflects the real position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The `choice` column from the trials table, which uses IBL convention: -1 (right), 0 (no choice), +1 (left).

ii.
```python
choice = final_trials['choice'].values.copy()
choice[choice == -1] = 0
choice = choice.astype(np.float32)
```

iii. The code maps choice values to binary output.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps `choice == -1` to 0 and keeps `choice == 1` as 1. The comment says "-1 (left) -> 0, 1 (right) -> 1" but IBL convention is `+1 = left, -1 = right`. The code effectively maps right choices to 0 and left choices to 1, which is the **opposite** of the instructions (left = 0, right = 1).

ii.
```python
# Choice: -1 (left) -> 0, 1 (right) -> 1    <-- comment has wrong IBL convention
choice = final_trials['choice'].values.copy()
choice[choice == -1] = 0   # -1 is rightward in IBL, mapped to 0
choice = choice.astype(np.float32)  # +1 (left) stays as 1
```

iii. The AI's comment incorrectly labels -1 as "left" when in IBL convention -1 is rightward. This results in an inverted choice encoding: left=1, right=0, instead of the specified left=0, right=1.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The `probabilityLeft` column from the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
prob_left = final_trials['probabilityLeft'].values.copy()
prior = np.zeros(len(prob_left), dtype=np.float32)
prior[np.isclose(prob_left, 0.2)] = 0
prior[np.isclose(prob_left, 0.5)] = 1
prior[np.isclose(prob_left, 0.8)] = 2
```

iii. The mapping follows the instructions: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct mapping using `np.isclose` to handle floating point comparison.

ii. Same as 6-a.

iii. No processing beyond the value mapping.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Raw wheel position (`_ibl_wheel.position.npy`) and timestamps (`_ibl_wheel.timestamps.npy`).

ii.
```python
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The AI loads raw wheel data files directly rather than using `SessionLoader`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI computes sampling frequency from median timestamp differences, applies Butterworth-filtered velocity using `velocity_filtered()` from brainbox, takes absolute value for speed, then interpolates to trial time bins using `np.linspace(t_beg + binsize, t_end, n_bins)`. Note: the AI does NOT first interpolate wheel position to a regular 1000 Hz grid (as `SessionLoader` does); instead it computes velocity directly from the irregularly-sampled position data.

ii.
```python
dt_median = np.median(np.diff(timestamps))
fs = 1.0 / dt_median
velocity, _ = velocity_filtered(position, fs)
speed = np.abs(velocity)
...
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
interp_func = interp1d(local_times, local_vals, kind='linear', fill_value='extrapolate')
```

iii. The AI attempts to match `SessionLoader.load_wheel` but skips the position interpolation step that SessionLoader performs first. The behavioral interpolation grid uses `np.linspace(t_beg + binsize, t_end, n_bins)` which differs from the reference's use of bin centres (`EDGES[:-1] + BIN/2`).

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI uses **global** discretization: all wheel speed values across all sessions are pooled, and the 33.33rd and 66.67th percentiles of the global distribution are used as bin edges. `np.digitize` is applied to assign values to 3 bins (0, 1, 2).

ii.
```python
all_wheel_vals = np.concatenate([w.flatten() for w in all_wheel_raw])
wheel_percentiles = np.percentile(all_wheel_vals[~np.isnan(all_wheel_vals)], [33.33, 66.67])
wheel_disc = np.digitize(wheel_raw, wheel_percentiles).astype(np.int64)
```

iii. The AI chose global percentiles to ensure consistent bin edges across sessions. The reference uses per-session percentiles.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated to time points computed as `np.linspace(t_beg + binsize, t_end, n_bins)` for each trial, where `t_beg` and `t_end` are absolute times. This gives N_BINS=100 values per trial.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. The interpolation grid has the same number of bins as the neural data, ensuring alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The motion energy from the left camera (`leftCamera.ROIMotionEnergy.npy`) with timestamps (`_ibl_leftCamera.times.npy`), falling back to right camera if left is unavailable.

ii.
```python
me_file = find_file(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_dir, '_ibl_leftCamera.times.npy')
if me_file is None or times_file is None:
    me_file = find_file(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_file(alf_dir, '_ibl_rightCamera.times.npy')
```

iii. Left camera preferred, right camera as fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are used as-is (no filtering or normalization). They are interpolated to the trial time bins using the same `interpolate_behavior_to_bins` function as wheel speed, then globally discretized into 3 bins.

ii.
```python
me_values = np.load(me_file).flatten()
me_times = np.load(times_file).flatten()
# Handle length mismatches and NaN removal
min_len = min(len(me_values), len(me_times))
me_values = me_values[:min_len]
me_times = me_times[:min_len]
valid = ~(np.isnan(me_values) | np.isnan(me_times))
```

iii. No additional processing beyond interpolation and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same global discretization as wheel speed: 33.33rd and 66.67th percentiles of all ME values across all sessions.

ii.
```python
all_me_vals = np.concatenate([m.flatten() for m in all_me_raw])
me_percentiles = np.percentile(all_me_vals[~np.isnan(all_me_vals)], [33.33, 66.67])
me_disc = np.digitize(me_raw, me_percentiles).astype(np.int64)
```

iii. Global percentiles for consistency across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated to `np.linspace(t_beg + binsize, t_end, n_bins)` for each trial.

ii.
```python
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, interval_begs, interval_ends, BINSIZE, N_BINS
)
```

iii. Same interpolation scheme as wheel speed ensures alignment with neural bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Trials with NaN in key columns are filtered out. (2) Sessions missing whisker ME data are skipped entirely. (3) Length mismatches between camera times and ME values are handled by truncation. (4) NaN values in ME data are removed. (5) Trials where behavioral interpolation fails are excluded via `combined_valid`. (6) Sessions with fewer than 2 valid trials are skipped.

ii.
```python
# Handle length mismatches
min_len = min(len(me_values), len(me_times))
me_values = me_values[:min_len]
me_times = me_times[:min_len]

# Remove NaN values
valid = ~(np.isnan(me_values) | np.isnan(me_times))

# Skip sessions with too few trials
if n_final < 2:
    return None
```

iii. The AI takes a conservative approach, dropping problematic data rather than imputing.

## 10-a. What are the most time-consuming steps of the code?

i. The spike binning is the most time-consuming per-session step. The `bin_spikes_fast` function loops over trials and uses `np.add.at` for accumulation, which is slower than `np.bincount`. Total conversion took ~37 minutes for 459 sessions.

ii.
```python
def bin_spikes_fast(spike_times, spike_clusters, n_clusters,
                    interval_begs, interval_ends, binsize, n_bins):
    for trial_idx in range(n_trials):
        ...
        linear_idx = trial_clusters * n_bins + bin_idx
        np.add.at(binned[trial_idx].ravel(), linear_idx, 1)
```

iii. The AI attempted to optimize by using searchsorted and vectorized indexing but the per-trial loop remains.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The behavioral interpolation loop (`interpolate_behavior_to_bins`) processes one trial at a time with `interp1d`, which has Python-level overhead. The spike binning loop could also be further vectorized by processing all trials at once.

ii.
```python
for trial_idx in range(n_trials):
    ...
    interp_func = interp1d(local_times, local_vals, kind='linear', fill_value='extrapolate')
    result[trial_idx] = interp_func(x_interp).astype(np.float32)
```

iii. Both loops iterate over trials, creating overhead from repeated Python function calls.

## 10-c. What processing does the code repeat multiple times?

i. The code has two separate spike binning implementations (`bin_spikes_vectorized` and `bin_spikes_fast`), though only `bin_spikes_fast` is called. The behavioral interpolation is called twice (once for wheel, once for whisker ME) with largely identical logic.

ii.
```python
def bin_spikes_vectorized(...)  # unused
def bin_spikes_fast(...)        # used
```

iii. The unused `bin_spikes_vectorized` function is dead code. The behavioral interpolation being called twice is necessary since it processes different data streams.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code includes an unused `bin_spikes_vectorized` function. It also stores raw wheel and ME values before discretization, which are only used for computing global percentiles and then discarded. The `show_processing` plotting code processes data that is only used for visualization.

ii.
```python
# Raw values stored temporarily
all_wheel_raw.append(result['wheel_speed_raw'])
all_me_raw.append(result['me_raw'])
```

iii. The temporary storage of raw behavioral values is necessary for the global discretization approach but would be unnecessary with per-session discretization.
