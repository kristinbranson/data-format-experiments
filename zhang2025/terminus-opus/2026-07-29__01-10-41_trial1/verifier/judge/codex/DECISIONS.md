# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not use the ONE API. It loads a release table from `code/code_zhang2025/data/bwm_release.csv`, groups rows by `eid`, finds each session under `data/one_cache/{lab}/Subjects/{subject}/{date}/001`, and then opens parquet and `.npy` files directly from disk with `pandas`, `numpy`, and `glob`.

ii.
```python
def find_session_dir(lab, subject, date, base_dir='data/one_cache'):
    session_dir = os.path.join(base_dir, lab, 'Subjects', subject, date, '001')
    if os.path.exists(session_dir):
        return session_dir
    return None
```

```python
bwm_df = pd.read_csv('code/code_zhang2025/data/bwm_release.csv', index_col=0)
session_groups = bwm_df.groupby('eid').agg({
    'lab': 'first',
    'subject': 'first',
    'date': 'first',
    'probe_name': list,
    'pid': list
}).reset_index()
```

iii. In `CONVERSION_NOTES.md` Step 2 and trajectory step 48, the agent says it can "load all the data directly from files" from the ONE cache layout and proceeds with direct file access rather than API-based loaders.

## 1-b. How are the data split into subjects?

i. Subjects come from the `subject` column of `bwm_release.csv`. During assembly, the agent creates subjects in first-seen order and records one `subject_idx` per processed session.

ii.
```python
session_groups = bwm_df.groupby('eid').agg({
    'lab': 'first',
    'subject': 'first',
    'date': 'first',
    'probe_name': list,
    'pid': list
}).reset_index()
```

```python
subject_map = {}
...
subj = result['subject']
if subj not in subject_map:
    subject_map[subj] = len(all_subjects)
    all_subjects.append(subj)
subject_idx_list.append(subject_map[subj])
```

iii. The notes describe the release table as session metadata and treat its `subject` field as the authoritative mouse identifier.

## 1-c. How are the data split into sessions?

i. Sessions are identified by `eid`. The agent groups the release table by `eid`, then processes each grouped row as one session.

ii.
```python
session_groups = bwm_df.groupby('eid').agg({
    'lab': 'first',
    'subject': 'first',
    'date': 'first',
    'probe_name': list,
    'pid': list
}).reset_index()
```

```python
for sess_idx, (_, row) in enumerate(session_groups.iterrows()):
    eid = row['eid']
    ...
    result = process_session(session_info, session_dir,
                             show_processing=args.show_processing)
```

iii. The notes state that `bwm_release.csv` contains 459 unique sessions and the code uses that session list as the unit of processing.

## 1-d. How are the data split into trials?

i. Trials come from rows of `_ibl_trials.table.pqt`. The agent loads the trial table into a dataframe and uses dataframe row indices as trial indices throughout masking and extraction.

ii.
```python
def load_trials(session_dir):
    trial_files = glob.glob(os.path.join(session_dir, 'alf', '*', '_ibl_trials.table.pqt'))
    if not trial_files:
        return None
    trials_df = pd.read_parquet(trial_files[0])
    return trials_df
```

```python
valid_idx = np.where(combined_mask)[0]
valid_trials = trials_df.iloc[valid_idx]
```

iii. The agent’s notes describe the trials table as containing trial-level variables directly, so no additional trial segmentation logic is introduced.

## 1-e. How are trials filtered based on quality controls?

i. The agent first builds a mask using reaction time bounds, maximum trial length, NaN exclusion on several fields, and exclusion of `choice == 0`. It then further requires wheel and whisker interpolation to succeed for the trial, which in practice means at least two non-NaN samples exist in the trial window for each stream.

ii.
```python
rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
if MIN_RT is not None:
    mask &= (rt >= MIN_RT)
if MAX_RT is not None:
    mask &= (rt <= MAX_RT)

trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
mask &= (trial_len <= MAX_TRIAL_LEN)

for event in nan_exclude:
    if event in trials_df.columns:
        mask &= ~trials_df[event].isna()

mask &= (trials_df['choice'] != 0)
```

```python
wheel_binned, wheel_valid = interpolate_behavior_to_bins(...)
me_binned, me_valid = interpolate_behavior_to_bins(...)
combined_mask = mask.values & wheel_valid & me_valid
```

iii. In `CONVERSION_NOTES.md` Step 1 and Step 3, the agent says it is matching `load_trials_and_mask()` from the reference utilities, and also adds behavior-availability filtering after interpolation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from per-spike times and per-spike cluster assignments, loaded from `spikes.times.npy` and `spikes.clusters.npy`. Cluster-channel and channel-brain-location files are used only to build neuron region labels.

ii.
```python
spike_times = np.load(os.path.join(ks_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(ks_dir, 'spikes.clusters.npy')).flatten()
```

```python
clusters_channels = np.load(os.path.join(ks_dir, 'clusters.channels.npy')).flatten()
...
brain_ids = np.load(brain_id_files[0]).flatten()
cluster_brain_ids = brain_ids[clusters_channels]
```

iii. The notes explicitly identify spikes as the neural source arrays and treat brain-region files as metadata needed for `brain_region_idx`.

## 2-b. How is the `neural` data processed?

i. The agent merges probes by offsetting cluster IDs, sorts all spikes by time, and bins spikes into 20 ms bins over each trial window. The final stored neural arrays are spike counts per cluster and bin; the code does not divide by bin width to convert to firing rates.

ii.
```python
spike_clusters = spike_clusters + cluster_offset
cluster_offset += n_clusters
...
sort_idx = np.argsort(merged_times, kind='stable')
merged_times = merged_times[sort_idx]
merged_clusters = merged_clusters[sort_idx]
```

```python
time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)
time_bin_idx = np.clip(time_bin_idx, 0, N_BINS - 1)
np.add.at(binned_all[trial_idx], (clust_curr, time_bin_idx), 1)
```

iii. The notes say the agent is following the reference binning setup (`binsize=0.02`, `time_window=(-0.5, 1.5)`) and the code comment describes the result as "spike counts" rather than rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered by neural quality control. All clusters found on each probe are kept, with no `label >= 1` filter and no exclusion of `void` regions.

ii.
```python
clusters_channels = np.load(os.path.join(ks_dir, 'clusters.channels.npy')).flatten()
n_clusters = len(clusters_channels)
...
all_cluster_regions.extend(beryl)
```

```python
if spike_times is None:
    print(f"  Skipping {eid}: no spike data")
    return None

n_clusters = len(cluster_regions)
```

iii. In `CONVERSION_NOTES.md` Step 1 and trajectory steps 8, 48, and 49, the agent states that the reference code uses `qc=None`, so it deliberately keeps all clusters.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial uses a window from `stimOn_times - 0.5` to `stimOn_times + 1.5`. Spike times are selected inside that interval and binned relative to the window start, so bin 0 corresponds to `-0.5 s` from stimulus onset and the 100 bins span stimulus-centered time.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

```python
stim_on = trials_df[ALIGN_TIME].values
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]
...
time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)
```

iii. In `CONVERSION_NOTES.md` Step 4, the agent explicitly resolves the alignment question by saying it will use `stimOn_times` for all variables because the task says to align to stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use fixed 20 ms bins over a 2 s window, giving 100 bins per trial. No later temporal rebinning is applied.

ii.
```python
BINSIZE = 0.02
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

```python
x_interp = np.linspace(t_beg, t_end, N_BINS, endpoint=False) + BINSIZE / 2
```

iii. The notes repeatedly cite the reference parameters `binsize=0.02` and `T=100`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the choice of `stimOn_times` as the alignment event plus the fixed decoder window and bin size. The code does not use a per-trial raw time series; it constructs the same relative-time vector for every trial.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
```

```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
time_since_stim = time_since_stim.astype(np.float32)
```

iii. The justification in Step 4 of the notes is that everything should be aligned to stimulus onset, so the time input is just the common stimulus-centered bin grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent computes bin centers from the fixed bin grid: start at `-0.5 s`, step by `20 ms`, and offset by `10 ms` to get centers.

ii.
```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
```

```python
inp = np.vstack([
    time_since_stim[np.newaxis, :],
    np.full((1, N_BINS), trial_num_in_block[i], dtype=np.float32)
])
```

iii. No separate justification is given beyond matching the 20 ms, stimulus-aligned reference window.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. It is exactly the same bin grid used to bin spikes and interpolate behaviors, so it is aligned by construction.

ii.
```python
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]
...
time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)
```

```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
```

iii. The notes say all streams will use `stimOn_times` and 20 ms bins, which makes this input the shared trial time axis.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft`. The code treats a change in `probabilityLeft` as the start of a new block.

ii.
```python
def compute_trial_number_in_block(prob_left):
    ...
    for i in range(len(prob_left)):
        if i > 0 and prob_left[i] != prob_left[i-1]:
            counter = 1
```

```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
```

iii. In trajectory step 62, the agent’s planning notes identify trial number in block as an input derived from block structure in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code scans trials in order, resets the counter whenever `probabilityLeft` changes, counts within block on all trials before filtering, and then selects the surviving trials. The count is 1-based, not 0-based.

ii.
```python
trial_nums = np.zeros(len(prob_left), dtype=np.float32)
counter = 1
for i in range(len(prob_left)):
    if i > 0 and prob_left[i] != prob_left[i-1]:
        counter = 1
    trial_nums[i] = counter
    counter += 1
```

```python
all_trial_nums = compute_trial_number_in_block(all_prob_left)
trial_num_in_block = all_trial_nums[valid_idx]
```

iii. The only explicit justification is the code comment "Need to compute on ALL trials first, then select valid ones," which preserves block position despite later trial drops.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the trials table.

ii.
```python
choice = valid_trials['choice'].values.copy()
choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
```

iii. In trajectory step 48, the agent notes that raw choice values are `-1, 0, 1` and says they need to be remapped for the decoder.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent excludes no-choice trials in the mask, then recodes `-1 -> 0` and everything else surviving the mask (effectively `+1`) to `1`. It documents this as left/right coding, but the implemented mapping is the inverse of the IBL convention used by the human reference.

ii.
```python
mask &= (trials_df['choice'] != 0)
```

```python
# Choice: -1 -> 0 (left), 1 -> 1 (right)
choice = valid_trials['choice'].values.copy()
choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
```

iii. The notes justify only that decoder choice must be binary; the trajectory shows the agent inferred the raw coding from data inspection and then chose this remapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trial-table column `probabilityLeft`.

ii.
```python
prob_left = valid_trials['probabilityLeft'].values
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
```

iii. The task specification directly gives the category mapping, and the agent repeats that mapping in its notes.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code performs a direct categorical remap `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`. Unexpected values are silently mapped to `1` by `dict.get`.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
```

iii. The justification is simply that this is the decoder output specification given in the instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
pos_file = os.path.join(session_dir, 'alf', '_ibl_wheel.position.npy')
ts_file = os.path.join(session_dir, 'alf', '_ibl_wheel.timestamps.npy')
...
wheel_pos = np.load(pos_file).flatten()
wheel_ts = np.load(ts_file).flatten()
```

iii. The notes explicitly identify wheel position and timestamps as the raw source, following the wheel-processing utilities from the reference stack.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel position is linearly interpolated to 1000 Hz, low-pass filtered with an order-8 Butterworth filter at 20 Hz, differentiated to velocity, converted to absolute speed, then interpolated onto trial bin centers.

ii.
```python
t_interp = np.arange(wheel_ts[0], wheel_ts[-1], 1.0 / WHEEL_FS)
pos_interp = interpolate.interp1d(wheel_ts, wheel_pos, kind='linear')(t_interp)
sos = signal.butter(N=WHEEL_FILTER_ORDER, Wn=WHEEL_CORNER_FREQ / WHEEL_FS * 2,
                    btype='lowpass', output='sos')
vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * WHEEL_FS
speed = np.abs(vel)
```

```python
wheel_binned, wheel_valid = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, trial_starts, trial_ends)
```

iii. In `CONVERSION_NOTES.md` Step 1 and trajectory step 56, the agent says it is reproducing `SessionLoader.load_wheel()`/`wheel.py` behavior directly because it is not using `SessionLoader`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The continuous wheel-speed values from all valid trials and time bins in a session are pooled, session-level quantile boundaries are computed for 3 equal-frequency bins, and `np.digitize` assigns categories `0, 1, 2`.

ii.
```python
def discretize_time_varying(values, n_bins=N_DISC_BINS):
    flat = values[~np.isnan(values)].flatten()
    ...
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    boundaries = np.quantile(flat, quantiles)
    result = np.digitize(values, boundaries).astype(np.int64)
```

```python
wheel_disc = discretize_time_varying(wheel_data, N_DISC_BINS)
```

iii. The notes say continuous decoder outputs need 3 bins, and the implementation chooses quantile-based equal-sized session bins.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is linearly interpolated onto the same 100 bin centers used for neural activity in the stimulus-centered trial window.

ii.
```python
x_interp = np.linspace(t_beg, t_end, N_BINS, endpoint=False) + BINSIZE / 2
...
result[trial_idx] = y_interp
```

```python
wheel_binned, wheel_valid = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, trial_starts, trial_ends)
```

iii. In Step 4 of the notes, the agent states that all variables will be aligned to stimulus onset, so wheel speed is placed on the neural bin grid rather than first-movement-centered bins.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` or `rightCamera.ROIMotionEnergy.npy` plus the matching camera time file. Left camera is preferred when present.

ii.
```python
me_file = find_file(session_dir, 'leftCamera.ROIMotionEnergy.npy')
cam_file = find_file(session_dir, '_ibl_leftCamera.times.npy')
...
me_file = find_file(session_dir, 'rightCamera.ROIMotionEnergy.npy')
cam_file = find_file(session_dir, '_ibl_rightCamera.times.npy')
```

```python
if me_file and cam_file:
    me = np.load(me_file).flatten()
    cam_times = np.load(cam_file).flatten()
```

iii. The notes explicitly say whisker motion energy comes from the released ROI motion energy trace and that left camera is tried first, matching the agent’s understanding of the reference code.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The agent uses the released motion-energy trace as-is, with no filtering or normalization, and linearly interpolates it to the trial bin centers.

ii.
```python
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, trial_starts, trial_ends)
```

```python
y_interp = interpolate.interp1d(
    beh_t_clean, beh_v_clean, kind='linear',
    fill_value='extrapolate')(x_interp)
```

iii. The notes describe whisker motion energy as a direct loaded behavior stream, unlike wheel speed which requires velocity computation.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is discretized the same way as wheel speed: pool all valid session values across trials and bins, compute 3 quantile boundaries, then digitize into `0, 1, 2`.

ii.
```python
me_disc = discretize_time_varying(me_data, N_DISC_BINS)
```

```python
boundaries = np.quantile(flat, quantiles)
result = np.digitize(values, boundaries).astype(np.int64)
```

iii. The agent’s notes say time-varying continuous decoder outputs should be discretized into 3 bins, and the code applies the same quantile rule to both outputs.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is interpolated onto the same stimulus-centered 20 ms bin centers used for neural activity.

ii.
```python
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]
...
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, trial_starts, trial_ends)
```

```python
x_interp = np.linspace(t_beg, t_end, N_BINS, endpoint=False) + BINSIZE / 2
```

iii. As with wheel speed, the explicit justification in the notes is that the task requires stimulus-onset alignment for all variables.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing files cause sessions or streams to be skipped. Missing wheel or whisker coverage marks trials invalid. Sessions with fewer than 2 valid trials are skipped. The code also falls back to `'unknown'` regions if brain-location files are missing and silently maps unexpected `probabilityLeft` values to class `1`.

ii.
```python
if not trial_files:
    return None
...
if not all_spike_times:
    return None, None, None
```

```python
if has_wheel:
    wheel_binned, wheel_valid = interpolate_behavior_to_bins(...)
else:
    wheel_valid = np.zeros(len(trials_df), dtype=bool)
...
combined_mask = mask.values & wheel_valid & me_valid
if n_valid < 2:
    print(f"  Skipping {eid}: only {n_valid} valid trials")
    return None
```

iii. The notes frame this as pragmatic data curation: keep only sessions with the required modalities, and drop trials whose aligned behavioral traces cannot be constructed.

## 10-a. What are the most time-consuming steps of the code?

i. The code treats per-session spike binning as a major cost: it prints timing specifically for the spike-binning phase and labels that routine an "optimized vectorized version." Behavior interpolation is another obvious repeated cost. File loading is also substantial but is not instrumented the same way.

ii.
```python
print(f"  Binning spikes ({n_clusters} clusters, {len(trials_df)} trials)...", end=' ')
t_bin = time.time()
binned_spikes = bin_spikes_per_trial(
    spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)
print(f"{time.time()-t_bin:.1f}s")
```

```python
def bin_spikes_per_trial(...):
    """... Optimized vectorized version."""
```

iii. The agent gives no separate prose analysis of runtime, but the code structure and comments show that spike binning was the performance focus.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Despite one comment calling the spike binning "vectorized," the code still loops over trials for spike binning and behavior interpolation. It also loops over trials again to build `input_list` and `output_list`, and loops once through all trials to compute trial numbers in block.

ii.
```python
for trial_idx in valid_indices:
    ...
    np.add.at(binned_all[trial_idx], (clust_curr, time_bin_idx), 1)
```

```python
for trial_idx in range(n_trials):
    ...
    y_interp = interpolate.interp1d(...)(x_interp)
```

```python
for i in range(n_valid_trials):
    inp = np.vstack([...])
    ...
for i in range(n_valid_trials):
    out = np.vstack([...])
```

iii. The agent’s justification is implicit: it seems to have optimized only enough to keep the code manageable while retaining per-trial loops.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several kinds of processing: repeated `glob`-based file discovery, repeated per-trial interpolation for wheel and whisker, repeated per-trial `np.vstack` assembly for inputs and outputs, and repeated file checks when selecting sample sessions.

ii.
```python
trial_files = glob.glob(os.path.join(session_dir, 'alf', '*', '_ibl_trials.table.pqt'))
...
spike_files = glob.glob(os.path.join(
    session_dir, 'alf', pname, 'pykilosort', '*', 'spikes.times.npy'))
...
matches = glob.glob(os.path.join(session_dir, 'alf', '*', filename))
```

```python
for i in range(n_valid_trials):
    inp = np.vstack([...])
    input_list.append(inp)
...
for i in range(n_valid_trials):
    out = np.vstack([...])
    output_list.append(out)
```

iii. There is no explicit justification beyond straightforward implementation convenience.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The largest unnecessary work is that spikes are binned for all trials before the trial mask is applied, and wheel/whisker traces are interpolated for all trials before invalid trials are discarded. This means expensive work is done on trials that never enter the saved dataset.

ii.
```python
# 7. Bin spikes per trial (for ALL trials first, then filter)
binned_spikes = bin_spikes_per_trial(
    spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)
```

```python
wheel_binned, wheel_valid = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, trial_starts, trial_ends)
...
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, trial_starts, trial_ends)
...
valid_idx = np.where(combined_mask)[0]
neural_data = binned_spikes[valid_idx]
```

iii. The code comment explicitly acknowledges this ordering ("for ALL trials first, then filter"), but no justification is given for why the discarded work is worth doing.
