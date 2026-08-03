# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not use the ONE API the way the human reference does. Instead, it loads cache index tables directly from disk (`sessions.pqt`, `datasets.pqt`, `bwm_release.csv`), intersects the release EIDs with the locally cached session table, resolves each session path inside `one_cache`, and then reads parquet and `.npy` files directly from those paths.

ii.
```python
sessions_df = pd.read_parquet(os.path.join(TABLES_DIR, 'sessions.pqt'))
datasets_df = pd.read_parquet(os.path.join(TABLES_DIR, 'datasets.pqt'))
bwm_df = pd.read_csv(BWM_CSV, index_col=0)

bwm_eids = bwm_df['eid'].unique()
cache_eids = set(sessions_df.index.astype(str))
available_eids = [eid for eid in bwm_eids if eid in cache_eids]
```

```python
def find_session_path(eid, sessions_df, datasets_df):
    if eid in datasets_df.index.get_level_values(0):
        session_path_str = datasets_df.loc[eid, 'session_path'].iloc[0]
        return Path(CACHE_DIR) / session_path_str
    if eid in sessions_df.index:
        row = sessions_df.loc[eid]
        lab = row['lab']
        subject = row['subject']
        date = str(row['date'])
        number = str(row['number']).zfill(3)
        return Path(CACHE_DIR) / lab / 'Subjects' / subject / date / number
```

iii. `CONVERSION_NOTES.md` justifies this in terms of using the local cache, stating that 354 of 459 sessions were available locally. I did not find a separate justification for abandoning the reference ONE/`SessionLoader`/`SpikeSortingLoader` loading path.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of `sessions_df` for each kept `eid`. The output `subjects` list is built in first-seen order as sessions are processed, and `subject_idx` stores the integer assigned at that first encounter.

ii.
```python
row = sessions_df.loc[eid]
subject = row['subject']
```

```python
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subject_to_idx)
    all_subjects.append(subject)
all_subject_idx.append(subject_to_idx[subject])
```

iii. There is no separate argument in the trajectory beyond using the session metadata already present in the cache tables.

## 1-c. How are the data split into sessions?

i. Sessions are split by `eid`. The script loops over `available_eids`, processes one `eid` at a time, and appends one session entry to `neural`, `input`, and `output` for each successful session.

ii.
```python
available_eids = [eid for eid in bwm_eids if eid in cache_eids]
...
for eid_idx, eid in enumerate(available_eids):
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
```

iii. The trajectory frames this as matching the release organization: one cached session per `eid`.

## 1-d. How are the data split into trials?

i. Trials are split row-wise from the `_ibl_trials.table.pqt` table. After initial trial masking, the script keeps `valid_trials = trials[mask]`, then applies a second mask for wheel and whisker coverage and iterates once per surviving trial.

ii.
```python
trials = pd.read_parquet(trials_file)
...
valid_trials = trials[mask].copy()
align_times = valid_trials[ALIGN_TIME].values
...
combined_indices = np.where(combined_mask)[0]
for trial_local_idx, trial_global_idx in enumerate(combined_indices):
    neural_trial = binned_spikes[trial_global_idx]
```

iii. No special justification was needed; the trajectory treats the trials table as already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The script first drops trials with NaNs in `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, or `feedbackType`, then excludes reaction times outside 0.08 to 2.0 s and no-choice trials (`choice == 0`). After that it discards trials whose wheel or whisker streams do not cover the full decoding window within a one-bin tolerance.

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']
...
for col in NAN_EXCLUDE:
    if col in trials.columns:
        mask &= ~trials[col].isna()
...
rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT)
mask &= (rt <= MAX_RT)
...
mask &= (trials['choice'] != 0)
```

```python
if np.abs(t_beg - local_times[0]) > binsize:
    good_mask[trial_idx] = False
...
if np.abs(t_end - local_times[-1]) > binsize:
    good_mask[trial_idx] = False
...
combined_mask = wheel_mask & me_mask
```

iii. `CONVERSION_NOTES.md` says these choices were meant to match `load_trials_and_mask()` in the reference code plus an extra quality check that behavioral data cover the full interval.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural arrays are built from `spikes.times.npy` and `spikes.clusters.npy`, with `clusters.channels.npy`, `clusters.metrics.pqt`, and `channels.brainLocationIds_ccf_2017.npy` used to remap cluster IDs, optionally quality-filter clusters, and assign brain regions.

ii.
```python
spike_times_file = find_file_with_revision(probe_dir, 'spikes.times.npy')
spike_clusters_file = find_file_with_revision(probe_dir, 'spikes.clusters.npy')
clusters_channels_file = find_file_with_revision(probe_dir, 'clusters.channels.npy')
clusters_metrics_file = find_file_with_revision(probe_dir, 'clusters.metrics.pqt')
channels_brain_ids_file = find_file_with_revision(probe_dir, 'channels.brainLocationIds_ccf_2017.npy')
```

```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
clusters_channels = np.load(clusters_channels_file).flatten()
channels_brain_ids = np.load(channels_brain_ids_file).flatten()
```

iii. The trajectory describes this as reproducing the spike-sorting inputs used by the reference code while loading them directly from cache.

## 2-b. How is the `neural` data processed?

i. The code merges probes from the same session, renumbers cluster IDs with offsets, sorts all spikes by time, and bins spikes into 20 ms bins in a `(-0.5, 1.5)` s window around `stimOn_times`. The resulting `neural` arrays are raw spike counts cast to `float64`; unlike the human reference, the code never divides by bin width to convert counts to Hz.

ii.
```python
for spike_times, spike_clusters, brain_ids, n_clusters in probes_data:
    all_spike_times.append(spike_times)
    all_spike_clusters.append(spike_clusters + cluster_offset)
    ...
sort_idx = np.argsort(merged_times, kind='stable')
merged_times = merged_times[sort_idx]
merged_clusters = merged_clusters[sort_idx]
```

```python
binned = np.zeros((n_trials, n_clusters_total, n_bins), dtype=np.float32)
...
bin_indices = np.minimum(
    ((trial_times - t_beg) / binsize).astype(np.int64),
    n_bins - 1
)
np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)
```

```python
neural_trial = binned_spikes[trial_global_idx]
session_neural.append(neural_trial.astype(np.float64))
```

iii. The notes justify the alignment window, 20 ms bins, and multi-probe merging as reference-matching decisions. I found no justification for leaving neural values as counts instead of converting them to firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. In code, each probe is filtered to clusters with `label >= 1.0` whenever `clusters.metrics.pqt` has a `label` column, and spikes from other clusters are dropped. Sessions are then skipped entirely if fewer than 5 clusters remain after merging probes. The top-level docstring and metadata still incorrectly claim that all clusters are included.

ii.
```python
def load_spike_data(session_path, probe_name, qc_threshold=1.0):
    ...
    if qc_threshold is not None and clusters_metrics_file is not None:
        metrics = pd.read_parquet(clusters_metrics_file)
        if 'label' in metrics.columns:
            good_mask = metrics['label'].values >= qc_threshold
            good_cluster_ids = np.where(good_mask)[0]
            ...
            spike_mask, ib = ismember(spike_clusters, good_cluster_ids)
            spike_times = spike_times[spike_mask]
            spike_clusters = ib.astype(np.int32)
```

```python
if n_clusters < MIN_NEURONS:
    print(f"  SKIP: too few clusters ({n_clusters} < {MIN_NEURONS})")
    n_skipped += 1
    continue
```

iii. The trajectory shows the agent first recognizing that the reference code used `qc=None`, then changing course because all-cluster output was too large. In step 98 it explicitly says it would filter to `label >= 1` "for data size and decoder accuracy," and `CONVERSION_NOTES.md` says this better matches the BWM paper's well-isolated-neuron analysis.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to `stimOn_times`. For each trial the script defines `t_beg = stimOn_times - 0.5` and `t_end = stimOn_times + 1.5`, selects spikes in that interval, and bins them relative to `t_beg`.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
t_beg = align_times[trial_idx] + time_window[0]
t_end = align_times[trial_idx] + time_window[1]
i_start = np.searchsorted(spike_times, t_beg, side='left')
i_end = np.searchsorted(spike_times, t_end, side='left')
```

iii. Both the trajectory and `CONVERSION_NOTES.md` justify this as directly matching the stated decoder task and the reference time window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Neural data are represented in 20 ms bins, 100 bins per trial. No further temporal rebinning is applied after spike binning.

ii.
```python
BINSIZE = 0.02
N_TIMEBINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

```python
bin_indices = np.minimum(
    ((trial_times - t_beg) / binsize).astype(np.int64),
    n_bins - 1
)
```

iii. The notes cite the methods paper and reference code for the 20 ms / 100-bin setting.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read as a raw data field. The script uses `stimOn_times` only to define the alignment window, then creates a synthetic time vector from `TIME_WINDOW` and `BINSIZE`.

ii.
```python
align_times = valid_trials[ALIGN_TIME].values
...
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
```

iii. `CONVERSION_NOTES.md` explicitly describes this input as a constructed time-varying signal ranging from `-0.48` s to `1.5` s.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code generates a linearly spaced 100-point vector from `-0.48` to `1.5` seconds and reuses it for every trial. This is intended as the time axis for the decoding window, but it is not the same as the reference bin centers.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
```

```python
input_trial_full = np.vstack([
    time_since_stim.reshape(1, -1),
    np.full((1, N_TIMEBINS), trial_num, dtype=np.float64)
])
```

iii. The notes justify this as "linearly spaced from -0.48s to 1.5s" and say it matches the behavior interpolation grid. There is no indication the agent noticed the 10 ms shift from the human reference bin centers.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The agent aligns this input by using the same synthetic grid that it also uses for behavioral interpolation, namely `linspace(t_beg + binsize, t_end, n_bins)`. That means the input is consistent with its behavior streams, but it is shifted relative to the true neural bin centers used by the human reference.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
...
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
```

iii. The notes explicitly describe the behavior interpolation and time input using that same grid and present this as the intended alignment rule.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft` in the trials table. A new block starts whenever `probabilityLeft` changes.

ii.
```python
prob_left = trials_df['probabilityLeft'].values
...
if prob_left[i] != prob_left[i-1] or np.isnan(prob_left[i]) or np.isnan(prob_left[i-1]):
    block_start = i
```

iii. `CONVERSION_NOTES.md` describes this as the 0-indexed count from block start, inferred from the block prior.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code loops across the full trials table, resets `block_start` when `probabilityLeft` changes, assigns each trial `i - block_start`, and only then applies the final trial mask. This means filtered-out trials still advance the count, matching the actual within-session block position.

ii.
```python
trial_nums = np.zeros(len(trials_df), dtype=np.float64)
block_start = 0
for i in range(1, len(prob_left)):
    if prob_left[i] != prob_left[i-1] or np.isnan(prob_left[i]) or np.isnan(prob_left[i-1]):
        block_start = i
    trial_nums[i] = i - block_start

return trial_nums[mask]
```

iii. The notes justify this as "0-indexed count from block start." No separate trajectory discussion was needed.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the `choice` column of the filtered trials table. The agent assumes the raw coding is `-1 = left` and `1 = right`, and excludes `choice == 0` beforehand.

ii.
```python
choice_vals = valid_trials['choice'].values  # -1 or 1
...
choice = 0 if choice_vals[trial_global_idx] == -1 else 1  # -1->0 (left), 1->1 (right)
```

iii. `CONVERSION_NOTES.md` explicitly states the same interpretation: "Original data: -1=left, 1=right."

## 5-b. What processing is involved in computing `output` *Choice*?

i. After no-choice trials are dropped, each surviving choice value is recoded to a binary constant trace for the whole trial: `-1 -> 0` and `1 -> 1`.

ii.
```python
if 'choice' in trials.columns:
    mask &= (trials['choice'] != 0)
```

```python
choice = 0 if choice_vals[trial_global_idx] == -1 else 1
output_trial = np.vstack([
    np.full((1, N_TIMEBINS), choice, dtype=np.int64),
    ...
])
```

iii. The notes frame this as the required left/right binary output, but they use the wrong raw-choice sign convention.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `probabilityLeft` in the filtered trials table.

ii.
```python
prob_left_vals = valid_trials['probabilityLeft'].values  # 0.2, 0.5, 0.8
```

iii. The notes describe this as the task prior carried by each block.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code maps `0.2 -> 0`, `0.5 -> 1`, and anything else to `2` (implicitly treating any remaining value as `0.8`). It then broadcasts that category across all time bins of the trial.

ii.
```python
prob = prob_left_vals[trial_global_idx]
if prob == 0.2:
    prior = 0
elif prob == 0.5:
    prior = 1
else:  # 0.8
    prior = 2
```

```python
np.full((1, N_TIMEBINS), prior, dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` says this follows the instructed 0.2/0.5/0.8 to 0/1/2 mapping.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from raw wheel position and timestamp arrays, `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
pos_file = alf_dir / '_ibl_wheel.position.npy'
ts_file = alf_dir / '_ibl_wheel.timestamps.npy'
...
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The notes say this matches the wheel-processing path used by the IBL tools.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The script interpolates wheel position to a uniform 1 kHz grid with `interpolate_position`, computes velocity with `velocity_filtered`, takes the absolute value, then linearly interpolates that speed trace onto one 100-point trial grid per trial.

ii.
```python
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
pos_interp, ts_interp = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(pos_interp, 1000)
speed = np.abs(velocity)
```

```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = interp1d(local_times, local_vals, kind='linear',
                   fill_value='extrapolate')(x_interp)
```

iii. `CONVERSION_NOTES.md` justifies this as matching IBL wheel preprocessing: interpolation to 1 kHz, velocity filtering, and absolute speed.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The agent concatenates all wheel-speed samples from surviving complete-data trials in a session, computes the 33.33rd and 66.67th percentiles, and discretizes each trial's time series with `np.digitize` into 3 categories.

ii.
```python
all_wheel_concat = np.concatenate(all_wheel_vals)
wheel_percentiles = np.percentile(all_wheel_concat, [33.33, 66.67])
...
wheel_disc = np.digitize(wheel_trial, wheel_percentiles).astype(np.int64)
```

iii. The notes justify this as per-session equal-frequency binning into low/medium/high categories.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned by interpolating onto `linspace(t_beg + binsize, t_end, n_bins)` for each trial. This is the same grid used for the time input, but it is shifted relative to the human reference neural bin centers.

ii.
```python
t_beg = align_times[trial_idx] + time_window[0]
t_end = align_times[trial_idx] + time_window[1]
...
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. `CONVERSION_NOTES.md` says this was intended to "match neural bin times" and explicitly names the same `linspace` grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` plus `_ibl_leftCamera.times.npy`, with fallback to the right camera files if the left-camera pair is unavailable.

ii.
```python
me_file = find_file_with_revision(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
ts_file = find_file_with_revision(alf_dir, '_ibl_leftCamera.times.npy')

if me_file is None or ts_file is None:
    me_file = find_file_with_revision(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
    ts_file = find_file_with_revision(alf_dir, '_ibl_rightCamera.times.npy')
```

iii. The notes explicitly justify left-camera preference with right-camera fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The code loads the released motion-energy trace, truncates the values and timestamps to equal length, drops NaNs, and linearly interpolates the result onto the same 100-point per-trial grid used for wheel speed. It does not apply additional smoothing or normalization.

ii.
```python
me = np.load(me_file).flatten()
times = np.load(ts_file).flatten()
min_len = min(len(me), len(times))
me = me[:min_len]
times = times[:min_len]
valid = ~np.isnan(me) & ~np.isnan(times)
...
return times[valid], me[valid]
```

```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = interp1d(local_times, local_vals, kind='linear',
                   fill_value='extrapolate')(x_interp)
```

iii. `CONVERSION_NOTES.md` describes this as using the released motion-energy signal directly, with only interpolation to the bin grid.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The script concatenates all whisker-motion-energy samples from complete-data trials in the session, computes the 33.33rd and 66.67th percentiles, and uses `np.digitize` to produce 3 categories.

ii.
```python
all_me_concat = np.concatenate(all_me_vals)
me_percentiles = np.percentile(all_me_concat, [33.33, 66.67])
...
me_disc = np.digitize(me_trial, me_percentiles).astype(np.int64)
```

iii. The notes justify this as equal-frequency low/medium/high discretization per session.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned using the same interpolation rule as wheel speed: `linspace(t_beg + binsize, t_end, n_bins)` in each trial. As with wheel speed, this is the agent's intended common decoder time grid, but it is shifted relative to the human reference neural bin centers.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = interp1d(local_times, local_vals, kind='linear',
                   fill_value='extrapolate')(x_interp)
```

iii. The notes explicitly describe this as interpolation to the neural bin times, using that `linspace` grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing files usually cause the current probe or session to be skipped by returning `None`. Missing or incomplete behavioral windows cause affected trials to be dropped. For whisker motion energy, unequal lengths are handled by truncating both arrays to the shorter length, and NaN samples are removed. Sessions are skipped if they have no valid probe data, too few valid trials, too few clusters, or too few complete-data trials.

ii.
```python
if trials_file is None:
    return None, None
...
if any(f is None for f in [spike_times_file, spike_clusters_file,
                            clusters_channels_file, channels_brain_ids_file]):
    return None, None, None, None
```

```python
min_len = min(len(me), len(times))
me = me[:min_len]
times = times[:min_len]
valid = ~np.isnan(me) & ~np.isnan(times)
if np.sum(valid) < 10:
    return None, None
```

```python
if n_valid_trials < 2:
    ...
if len(probes_data) == 0:
    ...
if n_clusters < MIN_NEURONS:
    ...
if n_final < 2:
    ...
```

iii. The trajectory and notes justify these mostly as pragmatic handling of incomplete cache contents and incomplete trial windows. I did not find a deeper justification for the extra session-level `MIN_NEURONS` cutoff.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading large spike arrays and per-cluster metadata from disk for every probe, then looping through every trial to bin spikes and interpolate behavior. The code itself flags spike binning as a costly step.

ii.
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
...
metrics = pd.read_parquet(clusters_metrics_file)
```

```python
print(f"  Binning spikes (this may take a moment)...")
binned_spikes = bin_spikes_per_trial(
    spike_times, spike_clusters, n_clusters,
    align_times, TIME_WINDOW, BINSIZE
)
```

iii. There is no explicit efficiency discussion in `CONVERSION_NOTES.md`, but the code comments and progress messages make clear that spike loading and spike binning were expected to dominate runtime.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The largest vectorization opportunities are the per-trial loops in `bin_spikes_per_trial()` and `interpolate_behavior_to_bins()`, the per-trial loop that assembles outputs, and the Python loop in `compute_trial_number_in_block()`. The agent already uses `np.add.at` inside each trial for spike counting, but not across trials.

ii.
```python
for trial_idx in range(n_trials):
    t_beg = align_times[trial_idx] + time_window[0]
    ...
    np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)
```

```python
for trial_idx in range(n_trials):
    ...
    y_interp = interp1d(local_times, local_vals, kind='linear',
                       fill_value='extrapolate')(x_interp)
```

```python
for i in range(1, len(prob_left)):
    ...

for trial_local_idx, trial_global_idx in enumerate(combined_indices):
    ...
```

iii. No explicit justification was given beyond clarity and directness of implementation.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly searches revision directories for individual files with `find_file_with_revision()`, separately interpolates wheel and whisker traces with the same logic, and separately concatenates all wheel and all whisker values to compute analogous percentiles. It also keeps both a generic `discretize_to_bins()` helper and hand-written percentile logic, but only the hand-written path is used.

ii.
```python
def find_file_with_revision(base_dir, filename):
    ...
```

```python
wheel_binned_list, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, TIME_WINDOW, BINSIZE
)
...
me_binned_list, me_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, TIME_WINDOW, BINSIZE
)
```

```python
all_wheel_concat = np.concatenate(all_wheel_vals)
all_me_concat = np.concatenate(all_me_vals)
wheel_percentiles = np.percentile(all_wheel_concat, [33.33, 66.67])
me_percentiles = np.percentile(all_me_concat, [33.33, 66.67])
```

iii. I did not find any explicit justification for these repeated paths.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest dead work is the unused helper `discretize_to_bins()`, the unused temporary `input_trial`, and the unused placeholders `wheel_binned` / `me_binned`. More broadly, the script builds continuous wheel and whisker traces only to discard them after percentile thresholding, because the final saved outputs retain only the discretized categories.

ii.
```python
def discretize_to_bins(values, n_bins=3):
    ...
```

```python
wheel_binned = None
...
me_binned = None
...
input_trial = np.array([trial_num], dtype=np.float64)  # (1,) per-trial
```

```python
wheel_trial = wheel_binned_list[trial_global_idx]
wheel_disc = np.digitize(wheel_trial, wheel_percentiles).astype(np.int64)
...
me_trial = me_binned_list[trial_global_idx]
me_disc = np.digitize(me_trial, me_percentiles).astype(np.int64)
```

iii. No explicit justification was documented; these look like leftovers from implementation rather than intentional retained processing.
