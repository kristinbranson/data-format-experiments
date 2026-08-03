# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the ONE API or `SessionLoader`/`SpikeSortingLoader` at runtime. Instead, it reads `code/code_zhang2025/data/bwm_release.csv`, groups rows by `eid`, resolves each session to a filesystem path under `data/one_cache/{lab}/Subjects/{subject}/{date}/001`, and then loads parquet and `.npy` files directly with `pandas`, `numpy`, and `glob`.

ii. 
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

```python
def find_session_dir(lab, subject, date, base_dir='data/one_cache'):
    session_dir = os.path.join(base_dir, lab, 'Subjects', subject, date, '001')
    if os.path.exists(session_dir):
        return session_dir
    return None
```

iii. In the trajectory, the AI explicitly decided to bypass the ONE/brainbox loaders because it was working offline and believed direct file loading from the cache was simpler and more reliable. `CONVERSION_NOTES.md` Step 2 also documents the cache-path layout as the basis for loading.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity comes from the `subject` column of `bwm_release.csv`. Sessions are processed one by one, and subjects are assigned integer indices in first-seen order using `subject_map`.

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

iii. The AI’s justification is implicit: the subject name is already present in the release CSV, so it reused that metadata rather than inferring subjects from paths.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in `bwm_release.csv`. The AI groups probe rows by `eid` to produce one session record with lab, subject, date, and probe list.

ii. 
```python
session_groups = bwm_df.groupby('eid').agg({
    'lab': 'first',
    'subject': 'first',
    'date': 'first',
    'probe_name': list,
    'pid': list
}).reset_index()
session_groups.rename(columns={'probe_name': 'probe_names'}, inplace=True)
```

iii. The justification is again implicit: the release table already represents the session/probe structure, so grouping by `eid` was treated as the natural session split.

## 1-d. How are the data split into trials?

i. Trials are taken directly from rows of the `_ibl_trials.table.pqt` parquet file. Later, only rows whose indices survive the combined mask are retained.

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

iii. The AI treated the trials table as already trial-structured, which matches how it described the dataset layout in `CONVERSION_NOTES.md`.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a trial mask based on reaction time, maximum trial length, NaNs in several fields, and nonzero choice. It then intersects that mask with wheel and whisker validity masks that require at least two samples per trial window for interpolation.

ii. 
```python
rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
if MIN_RT is not None:
    mask &= (rt >= MIN_RT)
if MAX_RT is not None:
    mask &= (rt <= MAX_RT)

trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
mask &= (trial_len <= MAX_TRIAL_LEN)
...
mask &= ~trials_df[event].isna()
...
mask &= (trials_df['choice'] != 0)
```

```python
if len(beh_v) < 2:
    valid[trial_idx] = False
...
combined_mask = mask.values & wheel_valid & me_valid
```

iii. `CONVERSION_NOTES.md` says this mask “matches `load_trials_and_mask()`” from the reference code, and the trajectory shows the AI believed the reference mask included `max_trial_len=10.0` and these NaN exclusions. It also decided to drop trials lacking interpolable wheel or motion-energy data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural tensor is derived from `spikes.times.npy` and `spikes.clusters.npy` loaded from each probe. Cluster-to-channel mappings and channel brain-location IDs are used only to attach region labels.

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

iii. The trajectory explicitly says the AI could “load all the data directly from files” and identified spike times and cluster assignments as the core neural inputs, with atlas mapping used for regions.

## 2-b. How is the `neural` data processed?

i. The AI merges spikes from all probes in a session by offsetting cluster IDs, sorts merged spikes by time, and bins spike counts into 20 ms bins for each trial window. It stores counts as `float32`; it does not divide by bin width to convert counts to firing rates.

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

iii. In `CONVERSION_NOTES.md` and `README.md`, the AI repeatedly describes the neural output as “spike counts binned at 20ms” and treats this as matching the reference cache format. There is no separate justification for not converting counts to Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered by cluster quality. All clusters found in `clusters.channels.npy` are kept, regardless of QC label.

ii. 
```python
clusters_channels = np.load(os.path.join(ks_dir, 'clusters.channels.npy')).flatten()
n_clusters = len(clusters_channels)
...
all_cluster_regions.extend(beryl)
```

iii. The trajectory and `CONVERSION_NOTES.md` both say the reference cache path used `qc=None`, so the AI concluded it should keep all clusters rather than applying `label >= 1` filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial windows are centered on `stimOn_times`, with `trial_starts = stimOn - 0.5` s and `trial_ends = stimOn + 1.5` s. Spikes are then binned relative to `t_beg`, so the neural bins are stimulus-onset aligned.

ii. 
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
stim_on = trials_df[ALIGN_TIME].values
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]
```

```python
time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)
```

iii. `CONVERSION_NOTES.md` Step 4 says the task specification required stimulus-onset alignment for all variables, and the AI explicitly chose to follow that interpretation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses 20 ms bins over a 2 s window, giving 100 time bins. No additional temporal rebinning is applied.

ii. 
```python
BINSIZE = 0.02
TIME_WINDOW = (-0.5, 1.5)
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The AI cites the reference decoder configuration in `CONVERSION_NOTES.md`: 2-second trials, 20 ms bins, 100 time steps.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from `stimOn_times` only indirectly: `stimOn_times` defines the aligned trial window, and the actual input values are synthetic bin-center times relative to stimulus onset.

ii. 
```python
stim_on = trials_df[ALIGN_TIME].values
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]
```

```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
time_since_stim = time_since_stim.astype(np.float32)
```

iii. The AI’s justification is that the decoder input is “time since stimulus onset,” so it constructed a fixed relative-time vector for every trial.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computes a length-100 vector of bin centers from `-0.49` s to `1.49` s using the global bin size and time window, then copies that same vector into every trial.

ii. 
```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
...
inp = np.vstack([
    time_since_stim[np.newaxis, :],
    np.full((1, N_BINS), trial_num_in_block[i], dtype=np.float32)
])
```

iii. No separate justification appears beyond the general decision to use the stimulus-aligned 20 ms decoder grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is placed on the same 100-bin stimulus-aligned grid as the neural data. The neural bins are computed from `trial_starts`, and the input uses the corresponding bin centers for that exact window.

ii. 
```python
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]
```

```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
```

iii. The AI explicitly framed the whole conversion around a common stimulus-onset time base in `CONVERSION_NOTES.md`.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the raw `probabilityLeft` sequence in the trials table. A block boundary is defined whenever `probabilityLeft` changes value.

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
trial_num_in_block = all_trial_nums[valid_idx]
```

iii. The trajectory says the AI understood block identity as recoverable from changes in `probabilityLeft`, since no explicit block ID was present.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI scans the full session’s `probabilityLeft` vector, resets a counter whenever the value changes, counts trials within a block starting from `1`, and only after that subsets to valid trials.

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

iii. The AI justified computing block counts before trial filtering in code comments and in the trajectory, so dropped trials still advance the block position.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the `choice` column of the trials table.

ii. 
```python
choice = valid_trials['choice'].values.copy()
```

iii. The trajectory explicitly notes the raw choice values are `-1`, `0`, and `1`, and that they must be remapped for the decoder.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI first removes no-choice trials through the mask, then recodes `-1` to `0` and all remaining non-`-1` values to `1`. In practice that means `-1 -> 0` and `+1 -> 1`.

ii. 
```python
mask &= (trials_df['choice'] != 0)
...
choice = valid_trials['choice'].values.copy()
choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
```

iii. In the trajectory, the AI explicitly stated “Need to map: `-1 -> left (0)`, `1 -> right (1)`,” and the code implements that exact interpretation.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `probabilityLeft` in the trials table.

ii. 
```python
prob_left = valid_trials['probabilityLeft'].values
```

iii. The AI’s notes repeatedly identify `probabilityLeft` as the block-prior variable that should become the decoder’s prior output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, using a default of `1` if an unexpected value appears.

ii. 
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
```

iii. The justification comes directly from the decoder task specification, which required this three-class remapping.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from raw wheel position and wheel timestamps: `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. 
```python
pos_file = os.path.join(session_dir, 'alf', '_ibl_wheel.position.npy')
ts_file = os.path.join(session_dir, 'alf', '_ibl_wheel.timestamps.npy')
...
wheel_pos = np.load(pos_file).flatten()
wheel_ts = np.load(ts_file).flatten()
```

iii. The trajectory and `CONVERSION_NOTES.md` both describe wheel speed as a derived behavioral trace computed from the wheel position stream using the IBL wheel-processing method.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI linearly interpolates wheel position to 1000 Hz, applies an 8th-order Butterworth low-pass filter at 20 Hz, differentiates to get velocity, takes absolute value to get speed, and then interpolates that speed trace onto the 100 per-trial bin centers.

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

iii. The AI explicitly justified this as matching the wheel-processing utilities in brainbox and described those exact steps in `CONVERSION_NOTES.md`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI discretizes wheel speed into three quantile-based bins computed over all non-NaN timepoints from the valid trials of a session.

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

iii. `README.md` and the sample-verification notes describe these as “3 quantile-based bins” and note that the class distribution is nearly even “by design.”

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed trace is interpolated onto the same 100 stimulus-aligned bin centers used for the neural data, one trial at a time.

ii. 
```python
x_interp = np.linspace(t_beg, t_end, N_BINS, endpoint=False) + BINSIZE / 2
y_interp = interpolate.interp1d(
    beh_t_clean, beh_v_clean, kind='linear',
    fill_value='extrapolate')(x_interp)
```

iii. The AI’s Step 4 notes say it chose a single stimulus-onset alignment for all variables, so wheel speed was forced onto that shared grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` or `rightCamera.ROIMotionEnergy.npy` plus the matching camera timestamps `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`, with the left camera preferred.

ii. 
```python
me_file = find_file(session_dir, 'leftCamera.ROIMotionEnergy.npy')
cam_file = find_file(session_dir, '_ibl_leftCamera.times.npy')
...
me_file = find_file(session_dir, 'rightCamera.ROIMotionEnergy.npy')
cam_file = find_file(session_dir, '_ibl_rightCamera.times.npy')
```

iii. The AI explicitly states in comments and notes that it is following the left-then-right camera preference from the reference behavior-loading logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI loads the released motion-energy trace without extra filtering, checks that it matches the camera timestamps, and interpolates it onto the 100 per-trial bin centers.

ii. 
```python
if me_file and cam_file:
    me = np.load(me_file).flatten()
    cam_times = np.load(cam_file).flatten()
    if len(me) == len(cam_times) and len(me) > 0:
        return cam_times, me
```

```python
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, trial_starts, trial_ends)
```

iii. `CONVERSION_NOTES.md` and `README.md` both describe whisker motion energy as being loaded from ROI motion-energy files and then interpolated to trial bins.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same three-way quantile discretization used for wheel speed, computed over all valid timepoints in that session’s motion-energy matrix.

ii. 
```python
me_disc = discretize_time_varying(me_data, N_DISC_BINS)
```

```python
boundaries = np.quantile(flat, quantiles)
result = np.digitize(values, boundaries).astype(np.int64)
```

iii. The AI justified this with the task requirement to make continuous outputs categorical and by mirroring the same “quantile-based bins” choice it used for wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is interpolated onto the same 100 stimulus-aligned bin centers as the neural data and wheel-speed output.

ii. 
```python
x_interp = np.linspace(t_beg, t_end, N_BINS, endpoint=False) + BINSIZE / 2
...
result[trial_idx] = y_interp
```

iii. The AI’s Step 4 notes say the task specification overrode any alternative alignment choices, so it aligned whisker motion energy to stimulus onset too.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handles missing or malformed data by skipping it. Missing files return `None`; sessions with no trials, no spikes, or fewer than two valid trials are dropped; wheel or motion-energy intervals with too few samples are marked invalid at the trial level; missing brain-region IDs fall back to `'unknown'`; unexpected prior values would silently map to class `1`.

ii. 
```python
if not trial_files:
    return None
...
if not all_spike_times:
    return None, None, None
...
if len(wheel_pos) != len(wheel_ts):
    return None, None
```

```python
if len(beh_v) < 2:
    valid[trial_idx] = False
...
if n_valid < 2:
    print(f"  Skipping {eid}: only {n_valid} valid trials")
    return None
```

```python
else:
    beryl = np.array(['unknown'] * n_clusters)
...
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
```

iii. The trajectory frames these choices as pragmatic handling for an offline cache with incomplete sessions. The AI repeatedly notes that some sessions lack wheel, motion-energy, or spike files and decides to skip such cases rather than repairing them.

## 10-a. What are the most time-consuming steps of the code?

i. The AI’s own execution logs show the dominant per-session cost is spike binning, especially for sessions with many clusters and trials. Behavior interpolation is secondary.

ii. 
```python
print(f"  Binning spikes ({n_clusters} clusters, {len(trials_df)} trials)...", end=' ')
t_bin = time.time()
binned_spikes = bin_spikes_per_trial(
    spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)
print(f"{time.time()-t_bin:.1f}s")
```

```python
wheel_binned, wheel_valid = interpolate_behavior_to_bins(...)
me_binned, me_valid = interpolate_behavior_to_bins(...)
```

iii. In the trajectory, the AI explicitly called out spike binning as “very slow” during sample runs and used those timings to motivate later optimization attempts.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization targets are the per-trial loops in `bin_spikes_per_trial` and `interpolate_behavior_to_bins`, plus the sequential loop in `compute_trial_number_in_block` and the per-neuron brain-region index assignment in the main assembly loop.

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
for i in range(len(prob_left)):
    if i > 0 and prob_left[i] != prob_left[i-1]:
        counter = 1
```

iii. The code labels the spike binning as an “Optimized vectorized version,” but the implementation still loops trial-by-trial. The trajectory also reflects the AI’s concern that spike binning remained a bottleneck.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats the same interpolation workflow separately for wheel and whisker traces; it computes trial windows and validity for every trial twice for those two modalities; and it repeatedly allocates constant per-trial time, choice, prior, and trial-number arrays instead of reusing shared templates.

ii. 
```python
wheel_binned, wheel_valid = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, trial_starts, trial_ends)
...
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, trial_starts, trial_ends)
```

```python
for i in range(n_valid_trials):
    inp = np.vstack([
        time_since_stim[np.newaxis, :],
        np.full((1, N_BINS), trial_num_in_block[i], dtype=np.float32)
    ])
```

```python
for i in range(n_valid_trials):
    out = np.vstack([
        np.full((1, N_BINS), choice_mapped[i], dtype=np.int64),
        np.full((1, N_BINS), prior_mapped[i], dtype=np.int64),
        wheel_disc[i:i+1, :],
        me_disc[i:i+1, :]
    ])
```

iii. There is no explicit defense of these repetitions. They follow from the AI’s choice to build every modality and every trial independently with straightforward loops.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The biggest discard is that spike counts are computed for all trials before trial filtering, and then only `valid_idx` trials are kept. The same is true for wheel and whisker interpolation. Sessions that later fail the combined mask still pay the full preprocessing cost. The code also builds dense `(n_trials, n_clusters, n_bins)` arrays before converting them back into per-trial Python lists.

ii. 
```python
binned_spikes = bin_spikes_per_trial(
    spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)
...
combined_mask = mask.values & wheel_valid & me_valid
valid_idx = np.where(combined_mask)[0]
neural_data = binned_spikes[valid_idx]
```

```python
result = np.full((n_trials, N_BINS), np.nan, dtype=np.float32)
...
wheel_data = wheel_binned[valid_idx]
me_data = me_binned[valid_idx]
```

iii. The trajectory shows the AI accepted this design because it was simpler to implement, even though many sessions and trials were later skipped.
