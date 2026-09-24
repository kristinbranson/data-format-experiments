# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local ONE cache tables directly, reads the Zhang BWM release CSV, intersects its unique `eid`s with cached session IDs, resolves each session path, and then opens trial, spike, wheel, and camera files directly. It processes 354 available sessions rather than selecting sessions by the complete required-dataset query used by the reference.

ii.
```python
sessions_df = pd.read_parquet(os.path.join(TABLES_DIR, 'sessions.pqt'))
datasets_df = pd.read_parquet(os.path.join(TABLES_DIR, 'datasets.pqt'))
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
bwm_eids = bwm_df['eid'].unique()
available_eids = [eid for eid in bwm_eids if eid in set(sessions_df.index.astype(str))]
```

iii. The trajectory says `SessionLoader` could not find revisioned trial tables, so the agent deliberately switched to direct file loading. It chose the Zhang release list because it found 354 of its 459 sessions in the cache.

## 1-b. How are the data split into subjects?

i. Each session's `subject` is read from `sessions_df`; subjects are assigned indices in first-processed-session order, and each retained session receives the corresponding index.

ii.
```python
subject = sessions_df.loc[eid]['subject']
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subject_to_idx)
    all_subjects.append(subject)
all_subject_idx.append(subject_to_idx[subject])
```

iii. The agent inspected the sessions table and recognized that it already supplies unique subject identifiers, so no path parsing was needed.

## 1-c. How are the data split into sessions?

i. One unique BWM `eid` is treated as one session and processed in the `available_eids` loop; outputs are appended only after all session checks pass.

ii.
```python
for eid_idx, eid in enumerate(available_eids):
    ...
    all_neural.append(session_neural)
```

iii. The trajectory identifies the cache/session table and unique `eid` as the natural session unit.

## 1-d. How are the data split into trials?

i. The trials parquet has one row per trial. Retained rows supply alignment times; spike and behavior streams are cut into the `[-0.5, 1.5)`-second window around each row's `stimOn_times`, and each retained row becomes one trial array.

ii.
```python
valid_trials = trials[mask].copy()
align_times = valid_trials[ALIGN_TIME].values
for trial_idx in range(n_trials):
    t_beg = align_times[trial_idx] + time_window[0]
    t_end = align_times[trial_idx] + time_window[1]
```

iii. The agent observed directly that the parquet table contains the needed trial columns and uses its rows as the trial definition.

## 1-e. How are trials filtered based on quality controls?

i. Trials with NaNs in listed event/task fields, reaction times outside 0.08–2 s, or `choice == 0` are removed. A second mask removes trials without wheel and motion-energy coverage; sessions with fewer than two final trials are skipped.

ii.
```python
for col in NAN_EXCLUDE:
    mask &= ~trials[col].isna()
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
mask &= (trials['choice'] != 0)
combined_mask = wheel_mask & me_mask
```

iii. The agent explicitly says these defaults match `load_trials_and_mask`; behavior coverage is required so every retained trial has every requested output.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays come from `spikes.times.npy` and `spikes.clusters.npy`. Cluster channel assignments, channel atlas IDs, and cluster metrics supply anatomical and QC information.

ii.
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
clusters_channels = np.load(clusters_channels_file).flatten()
channels_brain_ids = np.load(channels_brain_ids_file).flatten()
```

iii. The trajectory inspected the Zhang loaders and local probe files before choosing these arrays.

## 2-b. How is the `neural` data processed?

i. Probe populations are renumbered and merged, sorted by spike time, and spikes are counted per cluster in 100 20-ms bins. Unlike the reference, counts are not divided by 0.02 to yield Hz. Trial matrices are cast to float64.

ii.
```python
all_spike_clusters.append(spike_clusters + cluster_offset)
sort_idx = np.argsort(merged_times, kind='stable')
np.add.at(binned[trial_idx],
          (trial_clusters[valid], bin_indices[valid]), 1)
session_neural.append(neural_trial.astype(np.float64))
```

iii. The agent intended to follow the reference binning and probe merging. Its comments call the arrays spike counts; the trajectory discusses compactness but never justifies omitting conversion to firing rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Despite comments and metadata claiming all clusters are retained, the function defaults to `qc_threshold=1.0` and keeps metric labels at least 1. It does not remove Beryl `void` units and skips probes with no passing clusters. Sessions with fewer than five retained neurons are skipped.

ii.
```python
def load_spike_data(session_path, probe_name, qc_threshold=1.0):
    good_mask = metrics['label'].values >= qc_threshold
    good_cluster_ids = np.where(good_mask)[0]
...
if n_clusters < MIN_NEURONS:
    continue
```

iii. The trajectory first chose all clusters to match Zhang, then changed to label ≥1 because dense all-cluster output was estimated to be too large and the resulting unit counts better matched the data paper. This contradicts the final metadata string saying no filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute spikes are selected from 0.5 s before through 1.5 s after each `stimOn_times`, and bin indices are computed relative to that trial-window start.

ii.
```python
t_beg = align_times[trial_idx] + time_window[0]
t_end = align_times[trial_idx] + time_window[1]
bin_indices = ((trial_times - t_beg) / binsize).astype(np.int64)
```

iii. The agent identified stimulus onset and `(-0.5, 1.5)` as the Zhang choice-decoding configuration.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Spikes are newly binned into 100 bins of 20 ms over two seconds. There is no later neural resampling or rebinning.

ii.
```python
BINSIZE = 0.02
N_TIMEBINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The trajectory explicitly derives 100 bins from the reference configuration's 20-ms bin size and two-second window.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is a constructed common time grid defined by the window and bin size, with `stimOn_times` defining zero; it is not measured from another raw signal.

ii.
```python
ALIGN_TIME = 'stimOn_times'
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE,
                              TIME_WINDOW[1], N_TIMEBINS)
```

iii. The agent states that stimulus onset is the required alignment and constructs the decoder input from the reference window.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. `np.linspace` creates 100 values from -0.48 through 1.50 s. This represents right bin edges, not the reference bin centers (-0.49 through 1.49 s).

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_TIMEBINS
).astype(np.float64)
```

iii. The agent describes this as matching interpolation “bin centers,” but the implemented endpoints reveal an edge-grid interpretation.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same 100-value vector is stacked beside every trial's neural bins. Its indices correspond one-for-one, but each time label is 10 ms later than the neural bin center used by the reference.

ii.
```python
input_trial_full = np.vstack([
    time_since_stim.reshape(1, -1),
    np.full((1, N_TIMEBINS), trial_num)
])
```

iii. The agent intended identical alignment and dimensions across neural, input, and behavioral arrays.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the full unfiltered trial table's `probabilityLeft` sequence.

ii.
```python
prob_left = trials_df['probabilityLeft'].values
```

iii. The agent understood a block as a run over which the left prior is constant.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A counter starts at zero and resets whenever adjacent `probabilityLeft` values differ or either is NaN. It is computed before masking, then filtered, preserving original positions within blocks, and finally repeated across all time bins.

ii.
```python
if prob_left[i] != prob_left[i-1] or np.isnan(prob_left[i]) or np.isnan(prob_left[i-1]):
    block_start = i
trial_nums[i] = i - block_start
return trial_nums[mask]
```

iii. The agent's docstring says a change in `probabilityLeft` starts a new block; computing before filtering avoids renumbering after excluded trials.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes directly from the trial table's `choice` column, whose retained values are -1 and +1.

ii.
```python
choice_vals = valid_trials['choice'].values
```

iii. The agent inspected the column's unique values before implementing the required mapping.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Raw -1 is mapped to left/0 and every retained alternative (+1 after filtering) to right/1, then repeated over 100 bins.

ii.
```python
choice = 0 if choice_vals[trial_global_idx] == -1 else 1
np.full((1, N_TIMEBINS), choice, dtype=np.int64)
```

iii. This follows the requested left=0, right=1 encoding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from each trial row's `probabilityLeft` value.

ii.
```python
prob_left_vals = valid_trials['probabilityLeft'].values
```

iii. The agent inspected this column and found the expected 0.2, 0.5, and 0.8 values.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values are mapped 0.2→0, 0.5→1, and otherwise→2, then repeated over time. Earlier filtering ensures the `else` should mean 0.8.

ii.
```python
if prob == 0.2:
    prior = 0
elif prob == 0.5:
    prior = 1
else:
    prior = 2
```

iii. The mapping is copied from the task specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The trajectory traced Zhang's `wheel-speed` target to absolute wheel velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is interpolated at 1 kHz, filtered/differentiated with `velocity_filtered`, converted to absolute speed, linearly interpolated per trial, then discretized by session-wide percentiles.

ii.
```python
pos_interp, ts_interp = interpolate_position(timestamps, position, freq=1000)
velocity, _ = velocity_filtered(pos_interp, 1000)
speed = np.abs(velocity)
y_interp = interp1d(local_times, local_vals, kind='linear',
                    fill_value='extrapolate')(x_interp)
```

iii. The agent says this reproduces `SessionLoader.load_wheel` and the reference target definition.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Two thresholds are computed at 33.33 and 66.67 percent over all retained time samples in the session; `np.digitize` produces low/medium/high labels 0/1/2.

ii.
```python
wheel_percentiles = np.percentile(all_wheel_concat, [33.33, 66.67])
wheel_disc = np.digitize(wheel_trial, wheel_percentiles).astype(np.int64)
```

iii. The agent chose equal-frequency, session-specific bins, matching the reference in substance.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Each trace is interpolated at 100 points from window start +20 ms through window end, i.e. -0.48 through +1.50 s relative to stimulus. It has one value per neural bin but is shifted 10 ms from the reference's neural-bin centers.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
y_interp = interp1d(local_times, local_vals, kind='linear',
                    fill_value='extrapolate')(x_interp)
```

iii. The agent believed these points were the matching bin centers and used the common session clock.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses left-camera ROI motion energy and timestamps when present, with right-camera files as fallback.

ii.
```python
me_file = find_file_with_revision(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
ts_file = find_file_with_revision(alf_dir, '_ibl_leftCamera.times.npy')
```

iii. The agent identified this left-first fallback in the reference processing.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Arrays are truncated to equal length, NaN samples removed, linearly interpolated per trial, and discretized using session-wide percentiles. No filtering or normalization is applied.

ii.
```python
min_len = min(len(me), len(times))
valid = ~np.isnan(me) & ~np.isnan(times)
return times[valid], me[valid]
```

iii. The agent says released motion energy should be used directly apart from alignment and categorization; truncation/NaN removal handles imperfect files.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Thresholds are the session's 33.33rd and 66.67th percentiles over retained trials and times, producing labels 0–2 with `np.digitize`.

ii.
```python
me_percentiles = np.percentile(all_me_concat, [33.33, 66.67])
me_disc = np.digitize(me_trial, me_percentiles).astype(np.int64)
```

iii. Equal-frequency bins were selected to match the reference's three session-specific classes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera motion energy is interpolated to the same implemented -0.48…1.50-s grid as wheel output. Dimensions align with neural bins, but timestamps are 10 ms later than reference bin centers.

ii.
```python
me_binned_list, me_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, TIME_WINDOW, BINSIZE)
```

iii. The agent relies on synchronized session clocks and intended direct interpolation to neural-bin times.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Revision directories are searched; unavailable paths, trials, probes, wheel/camera streams, too-small sessions, NaNs, and incomplete behavior windows cause probes, trials, or sessions to be skipped. Motion arrays of unequal length are truncated, and invalid motion samples are removed.

ii.
```python
if result[0] is not None:
    probes_data.append(result)
...
if n_final < 2:
    continue
```

iii. The trajectory repeatedly tested local file layouts and designed direct revision lookup. It reports 31 of 354 sessions skipped for missing or insufficient usable data.

## 10-a. What are the most time-consuming steps of the code?

i. Loading large spike files and sequentially binning every session/trial dominate. The full single-worker conversion took tens of minutes; dense serialization also produces a roughly 20-GB pickle.

ii.
```python
for eid_idx, eid in enumerate(available_eids):
    ...
    binned_spikes = bin_spikes_per_trial(...)
```

iii. The agent called spike binning potentially slow, monitored a CPU-bound full conversion, and observed multi-gigabyte memory use. Although `--n-workers` exists, it is never used.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial loops in spike binning, behavior interpolation, per-trial value collection, and final trial assembly could be vectorized or reduced. The loop that computes trial number in block could use grouped cumulative counts.

ii.
```python
for trial_idx in range(n_trials):
    ...
for idx in combined_indices:
    ...
for trial_local_idx, trial_global_idx in enumerate(combined_indices):
```

iii. The trajectory praises vectorized `np.add.at` within a trial but does not discuss vectorizing the outer loops; practical performance concerns instead motivated neuron filtering.

## 10-c. What processing does the code repeat multiple times?

i. It scans/interpolates behavior separately for wheel and whisker, loops over retained trials once to collect arrays and again to build outputs, and repeatedly constructs identical time and constant per-trial rows. It also creates an unused `input_trial` variable.

ii.
```python
wheel_binned_list, wheel_mask = interpolate_behavior_to_bins(...)
me_binned_list, me_mask = interpolate_behavior_to_bins(...)
input_trial = np.array([trial_num], dtype=np.float64)
```

iii. No explicit justification for these repetitions appears in the trajectory; the implementation favors straightforward reusable helpers.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes acceleration and discards it, creates `input_trial` and never uses it, computes unused local variables (`wheel_binned`, `me_binned`, `trial_local_idx`), stores extensive metadata, casts sparse integer counts to float64, and runs distribution/sanity scans after assembly.

ii.
```python
velocity, _ = velocity_filtered(pos_interp, 1000)
input_trial = np.array([trial_num], dtype=np.float64)
for trial_local_idx, trial_global_idx in enumerate(combined_indices):
```

iii. The trajectory does not justify the dead variables. It did justify validation scans and metadata as sanity/documentation aids, although the decoder does not require them.
