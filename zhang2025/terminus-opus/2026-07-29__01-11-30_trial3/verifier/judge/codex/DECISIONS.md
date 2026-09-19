# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use the ONE API. It loaded session metadata from `code/code_zhang2025/data/bwm_release.csv`, scanned the local `data/one_cache` directory tree to build a `(subject, date) -> session_path` map, then opened individual parquet and `.npy` files directly from each session's `alf` folder.

ii. 
```python
DATA_DIR = 'data/one_cache'
BWM_CSV = 'code/code_zhang2025/data/bwm_release.csv'

bwm_df = pd.read_csv(BWM_CSV, index_col=0)
session_map = build_session_map(DATA_DIR)
```

```python
def build_session_map(data_dir):
    for lab_dir in os.listdir(data_dir):
        lab_path = os.path.join(data_dir, lab_dir, 'Subjects')
        ...
        key = (subject, date)
        session_map[key] = sess_path
```

```python
trials_df = pd.read_parquet(trials_file)
spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
position = np.load(pos_file).flatten()
me_values = np.load(me_file).flatten()
```

iii. The justification in `CONVERSION_NOTES.md` is that the cache structure was clear enough to read directly from disk, and the trajectory shows it decided to use `bwm_release.csv` plus direct file access after exploring the local cache rather than wiring up ONE.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column in `bwm_release.csv`. Sessions are accumulated subject by subject, and `subject_idx` is assigned in first-seen order among successfully processed sessions.

ii. 
```python
for eid, group in bwm_df.groupby('eid'):
    subject = group['subject'].iloc[0]
    ...
    sessions_info.append({
        'eid': eid,
        'subject': subject,
        ...
    })
```

```python
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(all_subjects)
    all_subjects.append(subject)
...
all_subject_idx.append(subject_to_idx[subject])
```

iii. The notes say the release CSV already provides subject IDs, so no extra derivation was needed.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid`s in `bwm_release.csv`, but the actual disk path is recovered through a `(subject, date)` lookup into the scanned cache tree. Probe names for a session come from grouping all rows with the same `eid`.

ii. 
```python
for eid, group in bwm_df.groupby('eid'):
    subject = group['subject'].iloc[0]
    date = group['date'].iloc[0]
    probe_names = list(group['probe_name'].unique())
    key = (subject, date)
    if key in session_map:
        sessions_info.append({...})
```

iii. The trajectory says it wanted to "match all 459 sessions" listed in the release CSV, and used the CSV as the authoritative session list while the scanned cache supplied paths.

## 1-d. How are the data split into trials?

i. Trials come from rows of `_ibl_trials.table.pqt`. After loading the full table, the AI applies a trial mask and then treats each surviving row as one trial.

ii. 
```python
def load_trials(alf_dir):
    trials_file = find_file(alf_dir, '_ibl_trials.table.pqt')
    trials_df = pd.read_parquet(trials_file)
    return trials_df
```

```python
mask = create_trials_mask(trials_df)
valid_trials_df = trials_df[mask].reset_index(drop=True)
```

iii. The notes explicitly say the trials table is one row per trial and should be used for trial segmentation.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept if they pass a Boolean mask built from reaction-time bounds, maximum trial length, non-null checks on several columns, and exclusion of no-choice trials while keeping unbiased (`probabilityLeft == 0.5`) trials. After that, trials are filtered again if wheel or whisker interpolation failed.

ii. 
```python
MIN_RT = 0.08
MAX_RT = 2.0
MAX_TRIAL_LEN = 10.0
EXCLUDE_UNBIASED = False
EXCLUDE_NOCHOICE = True
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']
```

```python
if MIN_RT is not None:
    query_parts.append(f'(firstMovement_times - stimOn_times < {MIN_RT})')
if MAX_RT is not None:
    query_parts.append(f'(firstMovement_times - stimOn_times > {MAX_RT})')
if MAX_TRIAL_LEN is not None:
    query_parts.append(f'(feedback_times - goCue_times > {MAX_TRIAL_LEN})')
...
if EXCLUDE_NOCHOICE:
    query_parts.append('(choice == 0)')
mask = ~trials_df.eval(query)
```

```python
wheel_binned, wheel_valid = interpolate_behavior_to_bins(...)
me_binned, me_valid = interpolate_behavior_to_bins(...)
combined_valid = wheel_valid & me_valid
```

iii. The main justification appears in `CONVERSION_NOTES.md` Step 4 and trajectory step 58: the AI believed these were the effective defaults of `load_trials_and_mask`, and it explicitly argued that unbiased trials had to be retained because prior `0.5` was required as a decoder output class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural arrays are derived from `spikes.times.npy` and `spikes.clusters.npy`, loaded probe by probe. The code also loads `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`, but those are used for region metadata rather than the neural time series itself.

ii. 
```python
spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(version_dir, 'spikes.clusters.npy')).flatten()
cluster_channels = np.load(os.path.join(version_dir, 'clusters.channels.npy')).flatten()
channel_brain_ids = np.load(os.path.join(version_dir, 'channels.brainLocationIds_ccf_2017.npy')).flatten()
```

iii. The notes and trajectory repeatedly identify spikes times/clusters as the core electrophysiology inputs, with channel/brain-location arrays needed to attach region identities.

## 2-b. How is the `neural` data processed?

i. Spike data from all probes in a session are merged, cluster IDs are offset so probes share one unit index space, spikes are sorted by time, and spikes are counted into 20 ms bins for each trial window. The resulting arrays are kept as spike counts in `float32`; the code does not convert them to firing rates.

ii. 
```python
all_spike_clusters.append(spike_clusters + cluster_offset)
cluster_offset += n_clusters
...
sort_idx = np.argsort(merged_times)
merged_times = merged_times[sort_idx]
merged_clusters = merged_clusters[sort_idx]
```

```python
bin_idx = np.minimum(
    ((trial_times - t_beg) / binsize).astype(np.int32),
    n_bins - 1
)
linear_idx = trial_clusters * n_bins + bin_idx
np.add.at(binned[trial_idx].ravel(), linear_idx, 1)
```

iii. The justification in the notes is that the reference code uses 20 ms bins over a 2 s window and merges probes recorded in the same session, so the AI followed that structure.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is effectively not filtered at all. The code loads every cluster present in the spike-sorting output and never consults cluster QC labels or removes `void` units.

ii. 
```python
def load_spikes(alf_dir, probe_name):
    ...
    return spike_times, spike_clusters, cluster_channels, channel_brain_ids
```

```python
spike_times, spike_clusters, cluster_brain_ids = merge_probes_data(probes_data)
n_clusters = len(cluster_brain_ids)
```

iii. The notes and trajectory explicitly justify this by saying the reference code used `qc=None`, so "all clusters" should be retained.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial window is centered on `stimOn_times`, with bins spanning `[-0.5, 1.5]` seconds relative to stimulus onset. Spikes are assigned to bins using per-trial `interval_begs = stim_on - 0.5` and `interval_ends = stim_on + 1.5`.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
stim_on = valid_trials_df[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

```python
i_start = np.searchsorted(spike_times, t_beg, side='left')
i_end = np.searchsorted(spike_times, t_end, side='left')
trial_times = spike_times[i_start:i_end]
bin_idx = np.minimum(((trial_times - t_beg) / binsize).astype(np.int32), n_bins - 1)
```

iii. The notes say stimulus onset was the required decoder alignment event and that the methods paper used the same `(-0.5, 1.5)` window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20 ms (`0.02` s) with 100 bins over a 2 s window. No second-stage temporal rebinning is applied.

ii. 
```python
BINSIZE = 0.02
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The notes justify this directly from the methods paper and reference code parameters.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read from a dedicated raw variable. It is constructed from the configured trial window and bin size, with `stimOn_times` providing the alignment event for each trial.

ii. 
```python
stim_on = valid_trials_df[ALIGN_TIME].values
...
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
).astype(np.float32)
```

iii. The notes describe this input as a derived coordinate on the stimulus-aligned bin grid rather than a separately stored measurement.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes a length-100 vector of nominal bin centers from `-0.49` to `1.49` seconds using `np.linspace`, then copies that same vector into every trial.

ii. 
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
).astype(np.float32)
```

```python
inp = np.stack([
    time_since_stim,
    np.full(N_BINS, trial_nums_final[t], dtype=np.float32)
], axis=0)
```

iii. The only justification in the notes is that the decoder needed a time-from-stimulus input on the same 20 ms grid as the neural data.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. It is intended to share the same stimulus-centered 100-bin grid as the neural data: one time vector is broadcast into each trial, and neural spikes are binned over the same `[-0.5, 1.5]` interval.

ii. 
```python
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
...
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
).astype(np.float32)
```

iii. The notes explicitly state that all decoder streams were to be aligned to stimulus onset on a common 20 ms grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the per-trial `probabilityLeft` column in the trials table. A block boundary is inferred whenever `probabilityLeft` changes value.

ii. 
```python
def compute_trial_number_in_block(prob_left):
    current_prob = prob_left[0]
    for i in range(len(prob_left)):
        if prob_left[i] != current_prob:
            current_block_start = i
            current_prob = prob_left[i]
        trial_nums[i] = i - current_block_start + 1
```

iii. The notes justify this by saying there is no explicit block index in the raw table, so the block structure must be reconstructed from `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code scans all trials in order, resets the block start when `probabilityLeft` changes, assigns the first trial of a block the value `1`, and only after that filters down to valid trials. The per-trial value is then broadcast across time bins.

ii. 
```python
trial_nums[i] = i - current_block_start + 1
```

```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
valid_indices = np.where(mask.values)[0]
trial_nums_valid = all_trial_nums[valid_indices]
trial_nums_final = trial_nums_valid[combined_valid]
```

iii. The notes explicitly say the count should be computed from all trials before filtering so dropped trials still advance the within-block counter. No justification was given for 1-based rather than 0-based indexing.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived directly from the trials-table `choice` column.

ii. 
```python
choice = final_trials['choice'].values.copy()
```

iii. The notes identify `trials_df.choice` as the source variable for the choice output.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code keeps only trials that survived the no-choice filter, then recodes `-1` to `0` and leaves `+1` as `1`. The accompanying comment states this means `-1` is left and `+1` is right.

ii. 
```python
# Choice: -1 (left) -> 0, 1 (right) -> 1
choice = final_trials['choice'].values.copy()
choice[choice == -1] = 0
choice = choice.astype(np.float32)
```

iii. The notes justify excluding `choice == 0` because the decoder output must be binary. The left/right sign convention comes from the AI's interpretation of the trials table and is reflected consistently in its comment and code.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trials-table `probabilityLeft` column.

ii. 
```python
prob_left = final_trials['probabilityLeft'].values.copy()
```

iii. The notes repeatedly identify `probabilityLeft` as the raw source for the prior output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code remaps the three allowed probabilities to categorical integers: `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`.

ii. 
```python
prior = np.zeros(len(prob_left), dtype=np.float32)
prior[np.isclose(prob_left, 0.2)] = 0
prior[np.isclose(prob_left, 0.5)] = 1
prior[np.isclose(prob_left, 0.8)] = 2
```

iii. The notes justify keeping `0.5` trials specifically because the decoder task requested it as its own class.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. 
```python
pos_file = find_file(alf_dir, '_ibl_wheel.position.npy')
ts_file = find_file(alf_dir, '_ibl_wheel.timestamps.npy')
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The notes and trajectory both state that wheel speed should come from wheel position plus timestamps and should match the reference behavior loader.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code estimates a sampling rate from the median timestamp difference, passes the raw position trace to `velocity_filtered`, takes the absolute value as speed, then linearly interpolates the result into each trial's 100 stimulus-aligned bins. If filtering fails, it falls back to a simple finite-difference velocity.

ii. 
```python
dt_median = np.median(np.diff(timestamps))
fs = 1.0 / dt_median
try:
    velocity, _ = velocity_filtered(position, fs)
except Exception:
    dt = np.diff(timestamps)
    ...
    velocity[1:] = np.diff(position) / dt
speed = np.abs(velocity)
```

```python
interp_func = interp1d(local_times, local_vals, kind='linear',
                       fill_value='extrapolate')
result[trial_idx] = interp_func(x_interp).astype(np.float32)
```

iii. The trajectory shows a late correction: after comparing a simple derivative against `velocity_filtered`, the AI concluded it needed the Butterworth-filtered version to better match the reference and patched the script accordingly.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is not discretized per session. Instead, the code pools wheel-speed values from all processed sessions, computes global 33.33rd and 66.67th percentiles, and then digitizes every time bin against those two global edges.

ii. 
```python
all_wheel_vals = np.concatenate([w.flatten() for w in all_wheel_raw])
wheel_percentiles = np.percentile(all_wheel_vals[~np.isnan(all_wheel_vals)], [33.33, 66.67])
```

```python
wheel_disc = np.digitize(wheel_raw, wheel_percentiles).astype(np.int64)
```

iii. The notes say the goal was "3 equal-frequency bins," but the code implements that globally across the dataset rather than session by session. No explicit justification for the global rather than per-session choice was found.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is interpolated separately within each trial onto 100 time points between `t_beg + binsize` and `t_end`, where `t_beg = stimOn - 0.5` and `t_end = stimOn + 1.5`. Trials with too few wheel samples in a padded search window are dropped.

ii. 
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
...
i_start = np.searchsorted(beh_times, t_beg - pad, side='left')
i_end = np.searchsorted(beh_times, t_end + pad, side='right')
...
result[trial_idx] = interp_func(x_interp).astype(np.float32)
```

iii. The notes justify aligning wheel output to the same stimulus-centered decoder window as the neural data, even though the original methods text discussed movement-centered decoding for wheel/whisker in a different context.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from either `leftCamera.ROIMotionEnergy.npy` with `_ibl_leftCamera.times.npy`, or, if the left camera files are missing, the corresponding right-camera files.

ii. 
```python
me_file = find_file(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_dir, '_ibl_leftCamera.times.npy')

if me_file is None or times_file is None:
    me_file = find_file(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_file(alf_dir, '_ibl_rightCamera.times.npy')
```

iii. The notes explicitly justify "left camera first, fall back to right" as following the reference code.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The code loads the released motion-energy trace directly, truncates it to the shorter of the values/times arrays if their lengths differ, removes NaN samples, and linearly interpolates the cleaned trace into each trial's 100 time bins.

ii. 
```python
min_len = min(len(me_values), len(me_times))
me_values = me_values[:min_len]
me_times = me_times[:min_len]

valid = ~(np.isnan(me_values) | np.isnan(me_times))
me_values = me_values[valid]
me_times = me_times[valid]
```

```python
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, interval_begs, interval_ends, BINSIZE, N_BINS
)
```

iii. The notes justify using the released motion-energy trace directly and mention only minimal cleanup for missing or mismatched data.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. As with wheel speed, the code uses global thresholds across all sessions: it concatenates every motion-energy value from every processed session, takes the 33.33rd and 66.67th percentiles, and digitizes all time bins with those edges.

ii. 
```python
all_me_vals = np.concatenate([m.flatten() for m in all_me_raw])
me_percentiles = np.percentile(all_me_vals[~np.isnan(all_me_vals)], [33.33, 66.67])
...
me_disc = np.digitize(me_raw, me_percentiles).astype(np.int64)
```

iii. The notes describe the target as 3 equal-frequency bins, but, as for wheel speed, they do not justify why the code switched from the reference session-wise discretization to one global discretization pass.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The whisker trace is aligned the same way as wheel speed: interpolated per trial onto 100 points spanning the stimulus-aligned window from `t_beg + binsize` to `t_end`, with trial removal if interpolation is not possible.

ii. 
```python
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, interval_begs, interval_ends, BINSIZE, N_BINS
)
...
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. The notes say all decoder outputs were intentionally forced onto the same stimulus-onset grid as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing or problematic data mostly by dropping data or falling back to simpler processing. Missing trials are excluded by the trial mask; sessions with no trials, no spikes, no wheel, or no whisker data are skipped; wheel filtering falls back to finite differences if `velocity_filtered` fails; mismatched whisker arrays are truncated; NaN whisker samples are removed; and trials where wheel/whisker interpolation fails are dropped.

ii. 
```python
if trials_df is None:
    return None
...
if n_valid < 2:
    return None
...
if len(probes_data) == 0:
    return None
...
if wheel_times is None:
    return None
...
if me_times is None:
    return None
```

```python
except Exception:
    dt = np.diff(timestamps)
    ...
    velocity[1:] = np.diff(position) / dt
```

```python
min_len = min(len(me_values), len(me_times))
...
valid = ~(np.isnan(me_values) | np.isnan(me_times))
```

iii. `CONVERSION_NOTES.md` says missing whisker sessions were skipped and that this was acceptable because those sessions could not supply the required decoder outputs. The trajectory shows similar reasoning for pragmatic fallbacks.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps are reading and merging the very large spike files for each probe/session, plus per-trial spike binning across all neurons and time bins. Saving the final pickle is also expensive because the output file is extremely large.

ii. 
```python
spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(version_dir, 'spikes.clusters.npy')).flatten()
```

```python
binned_spikes = bin_spikes_fast(
    spike_times, spike_clusters, n_clusters,
    interval_begs, interval_ends, BINSIZE, N_BINS
)
```

iii. The runtime notes in `CONVERSION_NOTES.md` and the trajectory repeatedly focus on spike loading/binning and on the long save time for the final ~100 GB pickle.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are still explicit per-trial loops: spike binning loops over trials, behavior interpolation loops over trials, trial-in-block computation loops over trials, and output assembly loops over trials and sessions.

ii. 
```python
for trial_idx in range(n_trials):
    ...
    np.add.at(binned[trial_idx].ravel(), linear_idx, 1)
```

```python
for trial_idx in range(n_trials):
    ...
    result[trial_idx] = interp_func(x_interp).astype(np.float32)
```

```python
for i in range(len(prob_left)):
    ...
for t in range(n_final_trials):
    ...
for sess_i in range(len(all_neural)):
    ...
```

iii. The AI even named one helper `bin_spikes_vectorized`, but the active implementation still retains several Python loops for simplicity.

## 10-c. What processing does the code repeat multiple times?

i. The biggest repeated processing is behavioral output construction: the code first interpolates and stores continuous wheel/motion-energy traces for every session, then does a second full-dataset pass to compute global thresholds, and then loops through every session and every trial again to discretize those traces and build the final `output` arrays.

ii. 
```python
all_wheel_raw.append(result['wheel_speed_raw'])
all_me_raw.append(result['me_raw'])
```

```python
all_wheel_vals = np.concatenate([w.flatten() for w in all_wheel_raw])
all_me_vals = np.concatenate([m.flatten() for m in all_me_raw])
```

```python
for sess_i in range(len(all_neural)):
    ...
    wheel_disc = np.digitize(wheel_raw, wheel_percentiles).astype(np.int64)
    me_disc = np.digitize(me_raw, me_percentiles).astype(np.int64)
    for t in range(n_trials):
        out = np.stack([...])
```

iii. No explicit justification for this two-pass structure was found beyond the implicit desire to use global discretization thresholds.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code keeps large continuous wheel and whisker arrays for all sessions even though the final dataset stores only discretized categories; it also defines unused helper/state such as `discretize_continuous`, `output_list_wheel`, and `output_list_me`. Those raw continuous arrays are only an intermediate for the later global discretization and are discarded from the saved dataset.

ii. 
```python
output_list_wheel = []
output_list_me = []
...
return {
    ...
    'wheel_speed_raw': final_wheel,
    'me_raw': final_me,
}
```

```python
def discretize_continuous(values, n_bins=3):
    ...
    return discretized, edges
```

```python
wheel_disc = np.digitize(wheel_raw, wheel_percentiles).astype(np.int64)
me_disc = np.digitize(me_raw, me_percentiles).astype(np.int64)
...
'output': all_output,
```

iii. There is no explicit justification for the extra retained raw arrays besides enabling the later global thresholding step.
