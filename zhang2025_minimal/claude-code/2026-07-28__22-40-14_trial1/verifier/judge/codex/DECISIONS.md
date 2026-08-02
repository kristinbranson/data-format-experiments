# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads session metadata from `sessions.pqt` and `datasets.pqt`, loads the BWM release table from `bwm_release.csv`, intersects BWM session IDs with locally cached session IDs, then iterates session-by-session through the local ONE cache. Within each session it reads the trials parquet table and the probe-specific spike files directly from disk rather than using the ONE API.

ii. 
```python
sessions_df = pd.read_parquet(os.path.join(TABLES_DIR, 'sessions.pqt'))
datasets_df = pd.read_parquet(os.path.join(TABLES_DIR, 'datasets.pqt'))
bwm_df = pd.read_csv(BWM_CSV, index_col=0)

bwm_eids = bwm_df['eid'].unique()
cache_eids = set(sessions_df.index.astype(str))
available_eids = [eid for eid in bwm_eids if eid in cache_eids]

for eid_idx, eid in enumerate(available_eids):
    session_path = find_session_path(eid, sessions_df, datasets_df)
    trials, mask = load_trials(session_path)
```

iii. In `CONVERSION_NOTES.md`, the agent says only 354 of 459 sessions were available in the local cache. The trajectory shows it intentionally chose an offline, direct-file-loading approach after exploring the cache layout.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the `subject` column in `sessions_df`. The converter keeps a unique `subjects` list and records one `subject_idx` per processed session.

ii. 
```python
row = sessions_df.loc[eid]
subject = row['subject']

if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subject_to_idx)
    all_subjects.append(subject)
all_subject_idx.append(subject_to_idx[subject])
```

iii. The notes describe the output as 107 subjects across the processed sessions, so the subject split is purely metadata-driven from the session table.

## 1-c. How are the data split into sessions?

i. Each unique BWM `eid` is treated as one session. All probes from that `eid` are merged, but trials remain grouped under that single session entry in `neural`, `input`, and `output`.

ii. 
```python
for eid_idx, eid in enumerate(available_eids):
    ...
    probe_dirs = sorted([d.name for d in alf_dir.iterdir()
                        if d.is_dir() and d.name.startswith('probe')])
    ...
    spike_times, spike_clusters, cluster_brain_ids = merge_probes_data(probes_data)
    ...
    all_neural.append(session_neural)
    all_input.append(session_input)
    all_output.append(session_output)
```

iii. The agent justifies probe merging in both `CONVERSION_NOTES.md` and the trajectory by citing the reference code’s `merge_probes()` and the rationale that probes from the same session are not statistically independent.

## 1-d. How are the data split into trials?

i. Trials come from rows of `_ibl_trials.table.pqt`. The script first applies a trial-quality mask, then keeps `valid_trials = trials[mask]`. After that it applies an additional `combined_mask` requiring both wheel and whisker data coverage, and each surviving row becomes one trial in the converted output.

ii. 
```python
trials, mask = load_trials(session_path)
valid_trials = trials[mask].copy()
align_times = valid_trials[ALIGN_TIME].values

combined_mask = wheel_mask & me_mask
combined_indices = np.where(combined_mask)[0]

for trial_local_idx, trial_global_idx in enumerate(combined_indices):
    neural_trial = binned_spikes[trial_global_idx]
    session_neural.append(neural_trial.astype(np.float64))
```

iii. The notes say trials were filtered once for trial QC and then again for complete behavioral coverage. The trajectory shows the agent viewed this as necessary to guarantee aligned neural and output arrays.

## 1-e. How are trials filtered based on quality controls?

i. The script excludes trials with missing required events, reaction times outside 0.08 to 2.0 s, and no-choice trials (`choice == 0`). It then removes any remaining trial that lacks wheel or whisker data across the full alignment window.

ii. 
```python
for col in NAN_EXCLUDE:
    if col in trials.columns:
        mask &= ~trials[col].isna()

rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT)
mask &= (rt <= MAX_RT)

if 'choice' in trials.columns:
    mask &= (trials['choice'] != 0)

combined_mask = wheel_mask & me_mask
```

iii. `CONVERSION_NOTES.md` explicitly cites `load_trials_and_mask()` for the NaN, RT, and no-choice filters, and says behavior trials are excluded if the signal does not cover the full interval.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from probe-level spike times and cluster assignments, with cluster-to-channel and channel-to-brain-location mappings. If QC filtering is enabled, the cluster `label` from `clusters.metrics.pqt` is also used.

ii. 
```python
spike_times_file = find_file_with_revision(probe_dir, 'spikes.times.npy')
spike_clusters_file = find_file_with_revision(probe_dir, 'spikes.clusters.npy')
clusters_channels_file = find_file_with_revision(probe_dir, 'clusters.channels.npy')
clusters_metrics_file = find_file_with_revision(probe_dir, 'clusters.metrics.pqt')
channels_brain_ids_file = find_file_with_revision(probe_dir, 'channels.brainLocationIds_ccf_2017.npy')

spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
clusters_channels = np.load(clusters_channels_file).flatten()
channels_brain_ids = np.load(channels_brain_ids_file).flatten()
```

iii. The notes frame this as direct use of the spike-sorting output plus Beryl atlas mapping. The trajectory does not show any alternative neural source being considered.

## 2-b. How is the `neural` data processed?

i. For each session, probe spike trains are loaded, optionally QC-filtered, merged across probes with cluster reindexing, time-sorted, and binned into per-trial spike-count matrices over a fixed window. The final neural tensor for each trial has shape `(n_neurons, 100)`.

ii. 
```python
for spike_times, spike_clusters, brain_ids, n_clusters in probes_data:
    all_spike_times.append(spike_times)
    all_spike_clusters.append(spike_clusters + cluster_offset)
    cluster_offset += n_clusters

sort_idx = np.argsort(merged_times, kind='stable')
merged_times = merged_times[sort_idx]
merged_clusters = merged_clusters[sort_idx]

binned_spikes = np.zeros((n_trials, n_clusters_total, n_bins), dtype=np.float32)
...
np.add.at(binned[trial_idx], (trial_clusters[valid], bin_indices[valid]), 1)
```

iii. The notes say this matches `merge_probes()` and `bin_spiking_data -> get_spike_data_per_interval()` in the reference code, with 20 ms bins over a 2 s trial window.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The actual implementation filters to “good” clusters with `label >= 1.0` from `clusters.metrics.pqt`, which the agent interprets as the BWM paper’s well-isolated neuron criterion. This is inconsistent with the script header and metadata string, which still claim no quality filter.

ii. 
```python
def load_spike_data(session_path, probe_name, qc_threshold=1.0):
    ...
    if qc_threshold is not None and clusters_metrics_file is not None:
        metrics = pd.read_parquet(clusters_metrics_file)
        if 'label' in metrics.columns:
            good_mask = metrics['label'].values >= qc_threshold
            good_cluster_ids = np.where(good_mask)[0]
```

iii. `CONVERSION_NOTES.md` says the agent deliberately chose well-isolated neurons to match the BWM paper and keep file sizes manageable. The trajectory shows it initially planned to use `qc=None` like Zhang’s code, then reversed that choice after seeing the dataset size.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to stimulus onset, using the `stimOn_times` column as the alignment event and extracting spikes from -0.5 s to +1.5 s around each trial’s stimulus onset.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)

align_times = valid_trials[ALIGN_TIME].values
...
t_beg = align_times[trial_idx] + time_window[0]
t_end = align_times[trial_idx] + time_window[1]
```

iii. Both the notes and the trajectory cite `0_data_caching.py` for this choice and repeat that the decoder task requested stimulus-onset alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use a 20 ms bin size, giving 100 bins over the 2 s window. The script bins raw spikes directly at that resolution; there is no second rebinning step afterward.

ii. 
```python
BINSIZE = 0.02
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_TIMEBINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The notes justify this from both the methods excerpt and the Zhang reference configuration.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from a trial-varying raw signal beyond the alignment definition itself. The series is constructed from the configured `TIME_WINDOW`, `BINSIZE`, and the fact that all trials are aligned to `stimOn_times`.

ii. 
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
```

iii. `CONVERSION_NOTES.md` describes this input as “linearly spaced from -0.48s to 1.5s,” with the implicit justification that it should match the neural bin centers.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The script creates a deterministic 100-sample vector of bin-center times shared by all trials. It does not inspect per-trial timestamps after alignment; it simply instantiates the same linearly spaced series for each trial.

ii. 
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)

input_trial_full = np.vstack([
    time_since_stim.reshape(1, -1),
    np.full((1, N_TIMEBINS), trial_num, dtype=np.float64)
])
```

iii. The notes justify this as a continuous, time-varying decoder input aligned to the binning scheme.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned by construction: the input uses the same `N_TIMEBINS`, the same 20 ms spacing, and the same window endpoints as the neural spike binning.

ii. 
```python
binned_spikes = bin_spikes_per_trial(
    spike_times, spike_clusters, n_clusters,
    align_times, TIME_WINDOW, BINSIZE
)

time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
)
```

iii. The notes explicitly state that behavioral interpolation and the time input are both matched to the neural bin centers.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column in the trials table. Block boundaries are inferred whenever `probabilityLeft` changes.

ii. 
```python
prob_left = trials_df['probabilityLeft'].values
...
if prob_left[i] != prob_left[i-1] or np.isnan(prob_left[i]) or np.isnan(prob_left[i-1]):
    block_start = i
trial_nums[i] = i - block_start
```

iii. The notes describe this as a “0-indexed count from block start,” and the trajectory does not show any more complicated block-state model being attempted.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The script walks through the full session trial sequence, resets the counter when `probabilityLeft` changes, computes a zero-based offset from the current block start, then applies the trial mask and repeats the scalar across all time bins within each retained trial.

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

iii. The justification in the notes is minimal; it mainly states the intended semantic meaning rather than citing a specific reference implementation.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived directly from the trials table’s `choice` column.

ii. 
```python
choice_vals = valid_trials['choice'].values  # -1 or 1
```

iii. The notes say the original coding is `-1=left, 1=right`, which the converter remaps to the decoder’s requested binary encoding.

## 5-b. What processing is involved in computing `output` *Choice*?

i. After no-choice trials are filtered out, each remaining trial’s `choice` is mapped from IBL coding to decoder coding: `-1 -> 0` for left and `1 -> 1` for right. The scalar label is then repeated across all 100 time bins.

ii. 
```python
choice = 0 if choice_vals[trial_global_idx] == -1 else 1

output_trial = np.vstack([
    np.full((1, N_TIMEBINS), choice, dtype=np.int64),
    ...
])
```

iii. The notes justify this as matching the decoder task specification exactly.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived directly from the `probabilityLeft` column of the trials table.

ii. 
```python
prob_left_vals = valid_trials['probabilityLeft'].values  # 0.2, 0.5, 0.8
```

iii. The notes cite the IBL block structure and state that the decoder output should use the requested three-class coding.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The raw `probabilityLeft` values are discretely remapped to class IDs: `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`. Like choice, the class is repeated over all time bins in the saved output.

ii. 
```python
prob = prob_left_vals[trial_global_idx]
if prob == 0.2:
    prior = 0
elif prob == 0.5:
    prior = 1
else:
    prior = 2
```

iii. The notes justify this as a direct implementation of the decoder task, not as a transformation found in the Zhang code.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from raw wheel position and timestamp arrays loaded from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. 
```python
pos_file = alf_dir / '_ibl_wheel.position.npy'
ts_file = alf_dir / '_ibl_wheel.timestamps.npy'

position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The notes explicitly state that wheel speed comes from wheel position plus timestamps, following the same upstream processing as IBL’s `SessionLoader`.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel position trace is interpolated to 1 kHz, converted to velocity with `velocity_filtered`, made nonnegative with `abs`, and then linearly interpolated onto the neural trial bins. This preserves a continuous wheel-speed trace until the final categorization step.

ii. 
```python
from brainbox.behavior.wheel import interpolate_position, velocity_filtered
pos_interp, ts_interp = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(pos_interp, 1000)
speed = np.abs(velocity)

wheel_binned_list, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, TIME_WINDOW, BINSIZE
)
```

iii. The notes justify this by citing the reference code’s `load_target_behavior(... 'wheel-speed')` and ibllib’s wheel-processing helpers.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. The script pools all wheel-speed values from all retained trials within a session, computes the 33.33rd and 66.67th percentiles, and uses those as per-session thresholds for three equal-frequency bins.

ii. 
```python
all_wheel_concat = np.concatenate(all_wheel_vals)
wheel_percentiles = np.percentile(all_wheel_concat, [33.33, 66.67])
...
wheel_disc = np.digitize(wheel_trial, wheel_percentiles).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says this was a task-driven discretization choice because the decoder output had to be categorical even though the source behavior is continuous.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to stimulus onset, not first movement onset. For each retained trial, the continuous wheel-speed signal is interpolated onto 100 bins spanning -0.5 to +1.5 s relative to `stimOn_times`, using the same bin centers as the neural data.

ii. 
```python
wheel_binned_list, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, TIME_WINDOW, BINSIZE
)

x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. The notes justify this by saying the decoder instructions required stimulus-onset alignment, even though the methods paper describes first-movement alignment for dynamic behaviors.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from precomputed ROI motion-energy arrays and their camera timestamps. The script prefers `leftCamera.ROIMotionEnergy.npy` plus `_ibl_leftCamera.times.npy`, and falls back to the corresponding right-camera files.

ii. 
```python
me_file = find_file_with_revision(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
ts_file = find_file_with_revision(alf_dir, '_ibl_leftCamera.times.npy')

if me_file is None or ts_file is None:
    me_file = find_file_with_revision(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
    ts_file = find_file_with_revision(alf_dir, '_ibl_rightCamera.times.npy')
```

iii. The notes say left camera is preferred to match the reference code and the data paper’s whisker-pad view, with right camera only as fallback.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The script loads the precomputed motion-energy signal, truncates it to match the timestamp length, removes NaNs, rejects traces with too few valid samples, and then linearly interpolates the remaining signal onto the neural bin centers for each trial.

ii. 
```python
min_len = min(len(me), len(times))
me = me[:min_len]
times = times[:min_len]

valid = ~np.isnan(me) & ~np.isnan(times)
if np.sum(valid) < 10:
    return None, None

return times[valid], me[valid]
```

iii. The notes justify this as using the IBL-provided motion-energy output directly, rather than recomputing motion energy from raw video.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, whisker motion energy is discretized per session using pooled values from all retained trials and percentile thresholds at 33.33% and 66.67%.

ii. 
```python
all_me_concat = np.concatenate(all_me_vals)
me_percentiles = np.percentile(all_me_concat, [33.33, 66.67])
...
me_disc = np.digitize(me_trial, me_percentiles).astype(np.int64)
```

iii. The notes explicitly describe this as a three-bin equal-frequency discretization added to satisfy the decoder task.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned to the same stimulus-onset-centered trial window as the neural data. Each trial is interpolated onto the shared 100-bin grid from -0.5 to +1.5 s relative to `stimOn_times`.

ii. 
```python
me_binned_list, me_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, TIME_WINDOW, BINSIZE
)

x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. The notes justify this as another task-required deviation from the methods paper’s first-movement alignment for dynamic behaviors.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data are mostly handled by exclusion. Missing files cause the session or probe to be skipped; NaNs in required trial events or behavior traces cause trial rejection; mismatched whisker array lengths are trimmed to the shorter length; too-short behavior traces and behavior intervals with insufficient coverage are also rejected.

ii. 
```python
if trials_file is None:
    return None, None

if any(f is None for f in [spike_times_file, spike_clusters_file,
                            clusters_channels_file, channels_brain_ids_file]):
    return None, None, None, None

min_len = min(len(me), len(times))
me = me[:min_len]
times = times[:min_len]

if len(local_vals) == 0 or np.any(np.isnan(local_vals)):
    good_mask[trial_idx] = False
    result.append(None)
```

iii. `CONVERSION_NOTES.md` calls out missing sessions in the cache and says trials are excluded if the behavior signal does not cover the full window. The trajectory shows the agent preferred skipping bad data over imputation.

## 12-a. What are the most time-consuming steps of the code?

i. The main expensive steps are the outer loop over hundreds of sessions, per-trial spike binning across all spikes and neurons, per-trial behavior interpolation for wheel and whisker signals, and writing the very large pickle at the end.

ii. 
```python
for eid_idx, eid in enumerate(available_eids):
    ...
    binned_spikes = bin_spikes_per_trial(...)
    ...
    wheel_binned_list, wheel_mask = interpolate_behavior_to_bins(...)
    me_binned_list, me_mask = interpolate_behavior_to_bins(...)
...
with open(output_path, 'wb') as f:
    pickle.dump(data, f)
```

iii. The trajectory explicitly notes that “the spike binning is very slow for large sessions” and later discusses file size as a practical bottleneck.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `bin_spikes_per_trial`, the per-trial loop in `interpolate_behavior_to_bins`, the loops that gather all wheel and whisker values for percentile computation, and the final per-trial assembly loop are the clearest vectorization targets.

ii. 
```python
for trial_idx in range(n_trials):
    ...

for idx in combined_indices:
    if wheel_binned_list[idx] is not None:
        all_wheel_vals.append(wheel_binned_list[idx])

for trial_local_idx, trial_global_idx in enumerate(combined_indices):
    ...
```

iii. The trajectory shows the agent already optimized part of spike counting with `np.add.at`, but it kept the higher-level trial loops intact.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly scans revision subdirectories to locate files, repeats nearly identical interpolation logic separately for wheel and whisker data, and performs repeated per-session list-to-array conversions while building outputs.

ii. 
```python
for d in sorted(base_dir.iterdir()):
    if d.is_dir() and d.name.startswith('#') and d.name.endswith('#'):
        candidate = d / filename

wheel_binned_list, wheel_mask = interpolate_behavior_to_bins(...)
me_binned_list, me_mask = interpolate_behavior_to_bins(...)
```

iii. The trajectory and notes do not dwell on these repeats, but the implementation clearly duplicates the same lookup and interpolation patterns across modalities and files.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter defines an unused helper (`discretize_to_bins`), creates an unused `input_trial` scalar before immediately replacing it with `input_trial_full`, and expands per-trial scalars such as choice and prior into full length-100 time series even though they are static within a trial.

ii. 
```python
def discretize_to_bins(values, n_bins=3):
    ...

trial_num = trial_nums_in_block[trial_global_idx]
input_trial = np.array([trial_num], dtype=np.float64)  # unused
input_trial_full = np.vstack([
    time_since_stim.reshape(1, -1),
    np.full((1, N_TIMEBINS), trial_num, dtype=np.float64)
])

output_trial = np.vstack([
    np.full((1, N_TIMEBINS), choice, dtype=np.int64),
    np.full((1, N_TIMEBINS), prior, dtype=np.int64),
    ...
])
```

iii. The trajectory does not justify these specific choices; they appear to be convenience decisions made to keep every input/output trial in a uniform `(variables, time)` format.
