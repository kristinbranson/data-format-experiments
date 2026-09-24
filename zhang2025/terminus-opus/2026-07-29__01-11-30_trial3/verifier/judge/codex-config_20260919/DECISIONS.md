# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `bwm_release.csv`, recursively constructs a local `(subject, date) -> session directory` map, groups release rows by `eid`, and directly loads parquet/NumPy files from each matched session. It attempts 459 release sessions and processes 444; it does not use ONE to resolve datasets.

ii.
```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
session_map = build_session_map(DATA_DIR)
for eid, group in bwm_df.groupby('eid'):
    key = (subject, date)
    if key in session_map:
        sessions_info.append({'eid': eid, 'path': session_map[key], ...})
```

iii. The notes say the cache contains 459/459 release sessions and identify its local ALF layout. The agent chose the release CSV as the authoritative session list and direct local access as a practical way to process the cache.

## 1-b. How are the data split into subjects?

i. Subject identifiers come from the `subject` column of the release CSV. As successful sessions are processed, subjects are assigned indices in first-seen order.

ii.
```python
subject = group['subject'].iloc[0]
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(all_subjects)
    all_subjects.append(subject)
all_subject_idx.append(subject_to_idx[subject])
```

iii. The notes treat the release table's subject field as the mouse identity and report 136 retained mice after sessions with missing motion energy were skipped.

## 1-c. How are the data split into sessions?

i. Rows of the release CSV are grouped by `eid`; probe names from all rows in an `eid` are collected. The local directory is found by subject and date.

ii.
```python
for eid, group in bwm_df.groupby('eid'):
    probe_names = list(group['probe_name'].unique())
    key = (subject, date)
```

iii. The notes identify `eid` as the session unit and aim to reproduce the 459 sessions in the data paper.

## 1-d. How are the data split into trials?

i. The trials parquet already has one row per trial. Valid rows supply stimulus onsets; fixed two-second windows around each onset define neural and behavioral trial arrays.

ii.
```python
trials_df = pd.read_parquet(trials_file)
valid_trials_df = trials_df[mask].reset_index(drop=True)
stim_on = valid_trials_df[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

iii. The agent followed the reference's stimulus-aligned, fixed-window trial representation.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded for reaction time outside 0.08–2 s, duration above 10 s, NaNs in six specified fields, or `choice == 0`. Unbiased trials are deliberately retained. After interpolation, trials are also removed if either behavior stream cannot be interpolated from at least two nearby samples.

ii.
```python
query_parts.append('(firstMovement_times - stimOn_times < 0.08)')
query_parts.append('(firstMovement_times - stimOn_times > 2.0)')
query_parts.append('(feedback_times - goCue_times > 10.0)')
for event in NAN_EXCLUDE:
    query_parts.append(f'{event}.isnull()')
query_parts.append('(choice == 0)')
combined_valid = wheel_valid & me_valid
```

iii. The notes say no-choice trials cannot support a binary choice target and that 0.5 trials must remain because the requested prior target explicitly has three classes. They attribute the RT, duration, and NaN rules to reference defaults.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays are derived from each probe's `spikes.times.npy` and `spikes.clusters.npy`. `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` provide the region for each cluster.

ii.
```python
spike_times = np.load(...'spikes.times.npy').flatten()
spike_clusters = np.load(...'spikes.clusters.npy').flatten()
cluster_channels = np.load(...'clusters.channels.npy').flatten()
channel_brain_ids = np.load(...'channels.brainLocationIds_ccf_2017.npy').flatten()
```

iii. The notes identify these as the same spike files used by the reference implementation.

## 2-b. How is the `neural` data processed?

i. Probe cluster IDs are offset and merged, spikes are time-sorted, and spikes are counted into 100 non-overlapping 20 ms bins per trial. The stored values remain spike counts (`float32`); they are not divided by bin width or smoothed.

ii.
```python
all_spike_clusters.append(spike_clusters + cluster_offset)
sort_idx = np.argsort(merged_times)
linear_idx = trial_clusters * n_bins + bin_idx
np.add.at(binned[trial_idx].ravel(), linear_idx, 1)
```

iii. The notes state that the reference bins spikes at 20 ms and uses all probes together. They describe the result as “spike counts” and do not justify omitting conversion to Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered: every sorted cluster in the selected probe directories is retained, including low-quality and `void` clusters.

ii.
```python
n_clusters = len(cluster_channels)
all_spike_times.append(spike_times)
all_spike_clusters.append(spike_clusters + cluster_offset)
```

iii. The agent concluded from `load_spiking_data(qc=None)` in the method code that all clusters should be retained. The notes explicitly record “No QC filtering.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial spans `stimOn_times - 0.5` through `stimOn_times + 1.5`; absolute spike timestamps in that interval are assigned bins relative to the interval beginning.

ii.
```python
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
bin_idx = ((trial_times - t_beg) / binsize).astype(np.int32)
```

iii. The notes cite the method paper's stimulus-onset alignment and −0.5 to +1.5 s window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20 ms, producing 100 bins over two seconds. Raw spikes are binned once; no further rebinning or smoothing is applied.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The notes cite the method paper's 20 ms, 100-step representation.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is constructed from the configured window and bin size around each trial's `stimOn_times`, rather than measured from another raw stream.

ii.
```python
stim_on = valid_trials_df[ALIGN_TIME].values
time_since_stim = np.linspace(-0.5 + BINSIZE/2, 1.5 - BINSIZE/2, N_BINS)
```

iii. The agent says this follows the required stimulus alignment and uses bin centers.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A common 100-value `float32` vector from −0.49 to 1.49 s is generated and copied into every trial input.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE/2,
                              TIME_WINDOW[1] - BINSIZE/2,
                              N_BINS).astype(np.float32)
```

iii. The notes describe it as a continuous time-varying input on the neural bin centers.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It has one value for each of the 100 neural bins and is broadcast identically to every trial. Its values denote neural bin centers.

ii.
```python
inp = np.stack([time_since_stim,
                np.full(N_BINS, trial_nums_final[t], dtype=np.float32)])
```

iii. The agent intended the input and neural matrices to share the same 20 ms grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the complete trials table's `probabilityLeft`; a probability change starts a new block.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
```

iii. The notes explain that probabilityLeft is constant within a block and that counting before filtering preserves the animal's true location in the block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code scans all trials, resets on probability changes, and assigns positions starting at 1. It then selects surviving trials and broadcasts each position across time.

ii.
```python
if prob_left[i] != current_prob:
    current_block_start = i
trial_nums[i] = i - current_block_start + 1
```

iii. The docstring explicitly chooses one-based numbering. The notes report observed values beginning at 1.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes directly from the trials table's `choice` column after no-response rows are filtered.

ii.
```python
choice = final_trials['choice'].values.copy()
```

iii. The notes identify trial choice as the requested binary target.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code maps raw −1 to 0 and leaves raw +1 as 1, then broadcasts the value through all 100 bins. Thus the implementation labels −1 as left and +1 as right.

ii.
```python
choice[choice == -1] = 0
choice = choice.astype(np.float32)
np.full(N_BINS, int(choice[t]), dtype=np.int64)
```

iii. The notes explicitly assert “−1->0 (left), 1->1 (right),” which is the rationale for this mapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from each trial's `probabilityLeft` value.

ii.
```python
prob_left = final_trials['probabilityLeft'].values.copy()
```

iii. The notes identify this column as the task's block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values close to 0.2, 0.5, and 0.8 are mapped to 0, 1, and 2, respectively, then broadcast through time.

ii.
```python
prior[np.isclose(prob_left, 0.2)] = 0
prior[np.isclose(prob_left, 0.5)] = 1
prior[np.isclose(prob_left, 0.8)] = 2
```

iii. This mapping is specified directly by the user and repeated in the notes.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
velocity, _ = velocity_filtered(position, fs)
speed = np.abs(velocity)
```

iii. The agent aimed to match `SessionLoader.load_wheel` and the reference definition of speed as absolute velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. A sampling frequency is estimated from median timestamp spacing; `velocity_filtered` is applied directly to the raw positions, with finite differences as fallback. Absolute velocity is linearly interpolated to 100 trial query points, with extrapolation permitted, then discretized globally.

ii.
```python
fs = 1.0 / np.median(np.diff(timestamps))
velocity, _ = velocity_filtered(position, fs)
interp_func = interp1d(local_times, local_vals, kind='linear', fill_value='extrapolate')
```

iii. The notes claim this matches the reference's filtered velocity and linear behavioral interpolation.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Two thresholds are the 33.33rd and 66.67th percentiles pooled over every retained session and time point. `np.digitize` creates low/medium/high classes.

ii.
```python
all_wheel_vals = np.concatenate([w.flatten() for w in all_wheel_raw])
wheel_percentiles = np.percentile(all_wheel_vals[~np.isnan(all_wheel_vals)], [33.33, 66.67])
wheel_disc = np.digitize(wheel_raw, wheel_percentiles)
```

iii. The agent chose “3 equal-frequency bins across all data,” intending consistent global class definitions.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is evaluated at 100 points from interval start +20 ms through interval end, i.e. −0.48 through +1.50 s relative to stimulus onset. Neural bins instead span intervals whose centers are −0.49 through +1.49 s.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
result[trial_idx] = interp_func(x_interp)
```

iii. The agent says these points match the reference behavior interpolation and considers them aligned to the neural time bins.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `leftCamera.ROIMotionEnergy.npy` and `_ibl_leftCamera.times.npy`, falling back to the corresponding right-camera arrays.

ii.
```python
me_file = find_file(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_dir, '_ibl_leftCamera.times.npy')
```

iii. The notes say left-first/right-fallback matches the reference logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Values and timestamps are truncated to equal lengths, rows with NaNs are removed, and the raw motion-energy trace is linearly interpolated (or extrapolated) to trial query points before global discretization.

ii.
```python
min_len = min(len(me_values), len(me_times))
valid = ~(np.isnan(me_values) | np.isnan(me_times))
interp_func = interp1d(local_times, local_vals, kind='linear', fill_value='extrapolate')
```

iii. The agent states that released motion energy needs no additional filtering and that linear interpolation matches the reference.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The 33.33rd and 66.67th percentiles pooled over all retained sessions/time points define the three classes.

ii.
```python
all_me_vals = np.concatenate([m.flatten() for m in all_me_raw])
me_percentiles = np.percentile(all_me_vals[~np.isnan(all_me_vals)], [33.33, 66.67])
me_disc = np.digitize(me_raw, me_percentiles)
```

iii. The notes justify global equal-frequency classes as the planned discretization strategy.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Like wheel speed, it is interpolated at −0.48 through +1.50 s around `stimOn_times`, one sample per neural bin but shifted 10 ms later than neural bin centers.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
result[trial_idx] = interp_func(x_interp).astype(np.float32)
```

iii. The agent intended to reproduce the reference interpolation and considered the 100 output samples aligned with the 100 neural bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing trials files, probes, wheel, or camera streams cause a session to be skipped. Missing trial fields are filtered. Camera length mismatches are truncated and NaNs removed. Versioned files use the lexicographically latest match. Behavioral trial interpolation requires two nearby samples but permits extrapolation. Sessions with fewer than two surviving trials are skipped; file-loading exceptions are printed and treated as missing probes.

ii.
```python
if trials_df is None: return None
min_len = min(len(me_values), len(me_times))
valid = ~(np.isnan(me_values) | np.isnan(me_times))
if i_end - i_start < 2:
    valid_mask[trial_idx] = False
```

iii. The notes report that 15 sessions were skipped for absent whisker motion energy and that the remaining structure passed format verification.

## 10-a. What are the most time-consuming steps of the code?

i. Loading/merging tens of millions of spikes per session, per-trial spike binning, holding/writing the 99 GB result, and the final all-data concatenation for global behavioral percentiles dominate. The full conversion took about 34 minutes; large sessions spent several seconds binning.

ii.
```python
spike_times = np.load(...)
for trial_idx in range(n_trials):
    ...
    np.add.at(binned[trial_idx].ravel(), linear_idx, 1)
pickle.dump(data, f)
```

iii. The notes estimate roughly 2.5 seconds per session and about 21 minutes before the full run; the recorded run took 2031.7 seconds.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike binning loops over trials, output construction loops over sessions and trials, block numbering loops over trials, and region remapping uses Python loops. The unused alternative `bin_spikes_vectorized` additionally loops over every spike. Most could be replaced by grouped/vectorized indexing, though trial-window slicing makes a per-trial loop understandable.

ii.
```python
for trial_idx in range(n_trials):
    ...
for spike_i in range(len(trial_times)):
    binned[trial_idx, c, b] += 1
for t in range(n_trials):
    out = np.stack([...])
```

iii. The agent labels `bin_spikes_fast` “vectorized” and uses search/slice plus `np.add.at`, indicating an explicit efficiency effort, but does not discuss remaining loop-vectorization opportunities.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly searches the filesystem with glob for every dataset, repeatedly computes trial query grids and constructs interpolation objects for every behavior/trial, and later loops over all sessions/trials again to create outputs because discretization is deferred globally. Large behavioral arrays are flattened and concatenated after already being retained.

ii.
```python
for trial_idx in range(n_trials):
    x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
    interp_func = interp1d(local_times, local_vals, ...)
for sess_i in range(len(all_neural)):
    for t in range(n_trials):
        out = np.stack([...])
```

iii. No explicit rationale is given; the two-pass output construction is a consequence of the chosen global thresholds.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads and time-sorts all spikes, including spikes outside retained trial windows and clusters that reference QC would discard. It loads/maps region information for all those clusters. It defines an unused slower spike-binning function and imports/accepts plotting-related options during normal conversion; raw behavioral arrays and metadata are retained until global discretization and then discarded.

ii.
```python
spike_times = np.load(...'spikes.times.npy').flatten()
sort_idx = np.argsort(merged_times)
def bin_spikes_vectorized(...): ...  # never called
all_wheel_raw.append(result['wheel_speed_raw'])
```

iii. The notes justify retaining all clusters as reference-consistent and global raw traces as necessary for pooled thresholds, but do not identify these as avoidable work.
