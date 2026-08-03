# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads session metadata from `bwm_release.csv`, groups by `eid` to get session-level info with probe names, then iterates over sessions. For each session, it locates the session directory in the ONE cache (`data/one_cache/{lab}/Subjects/{subject}/{date}/001/`), loads trials from parquet files, spikes from numpy files, wheel data from numpy files, and whisker motion energy from numpy files. Sessions without complete data (missing spikes, trials, wheel, or motion energy) are skipped.

ii.
```python
# Load session info from bwm_release.csv
bwm_df = pd.read_csv('code/code_zhang2025/data/bwm_release.csv', index_col=0)
session_groups = bwm_df.groupby('eid').agg({
    'lab': 'first', 'subject': 'first', 'date': 'first',
    'probe_name': list, 'pid': list
}).reset_index()

# For each session:
trials_df = load_trials(session_dir)  # parquet
spike_times, spike_clusters, cluster_regions = load_spikes(session_dir, probe_names)  # numpy
wheel_times, wheel_speed = load_wheel_speed(session_dir)  # numpy + processing
me_times, me_values = load_whisker_motion_energy(session_dir)  # numpy
```

iii. The agent identified `bwm_release.csv` from the reference code's `0_data_caching.py` as the session manifest. It chose to load data directly from files rather than using the IBL ONE API (which was unavailable in the offline environment). The agent noted 459 sessions in bwm_release, with 340 ultimately processed (119 skipped due to missing data).

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `subject` column in `bwm_release.csv`. A `subject_map` dictionary tracks unique subjects and assigns indices. Each session's subject is looked up and an index into the `subjects` list is stored in `subject_idx`.

ii.
```python
subject_map = {}
# ...
subj = result['subject']
if subj not in subject_map:
    subject_map[subj] = len(all_subjects)
    all_subjects.append(subj)
subject_idx_list.append(subject_map[subj])
```

iii. The agent followed the standard convention from the reference code where subjects are identified by name from the session metadata.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values in `bwm_release.csv`. Each eid corresponds to one recording session. Multiple probes within the same session are grouped together. The main loop iterates over unique sessions.

ii.
```python
session_groups = bwm_df.groupby('eid').agg({
    'lab': 'first', 'subject': 'first', 'date': 'first',
    'probe_name': list, 'pid': list
}).reset_index()

for sess_idx, (_, row) in enumerate(session_groups.iterrows()):
    eid = row['eid']
    # ... process session
```

iii. The agent correctly identified that bwm_release.csv has one row per probe, and grouped by eid to get session-level records with lists of probe names.

## 1-d. How are the data split into trials?

i. Trials are loaded from `_ibl_trials.table.pqt` parquet files. Each row in the trials dataframe represents one trial. Trials are aligned to `stimOn_times` with a window of (-0.5, 1.5) seconds.

ii.
```python
def load_trials(session_dir):
    trial_files = glob.glob(os.path.join(session_dir, 'alf', '*', '_ibl_trials.table.pqt'))
    trials_df = pd.read_parquet(trial_files[0])
    return trials_df

# Trial intervals aligned to stimOn
stim_on = trials_df[ALIGN_TIME].values
trial_starts = stim_on + TIME_WINDOW[0]  # -0.5
trial_ends = stim_on + TIME_WINDOW[1]    # 1.5
```

iii. The agent followed the reference code's approach of using `stimOn_times` as the alignment event with a 2-second window.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using a mask that excludes: (1) trials with reaction time < 0.08s or > 2.0s, (2) trials with trial length > 10.0s (feedback_times - goCue_times), (3) trials with NaN in key fields (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), and (4) no-choice trials (choice == 0). Additionally, trials where wheel speed or whisker motion energy interpolation fails are excluded via a combined mask.

ii.
```python
def create_trials_mask(trials_df):
    nan_exclude = ['stimOn_times', 'choice', 'feedback_times',
                   'probabilityLeft', 'firstMovement_times', 'feedbackType']
    mask = pd.Series(True, index=trials_df.index)
    rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
    mask &= (rt >= MIN_RT)    # 0.08
    mask &= (rt <= MAX_RT)    # 2.0
    trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
    mask &= (trial_len <= MAX_TRIAL_LEN)  # 10.0
    for event in nan_exclude:
        if event in trials_df.columns:
            mask &= ~trials_df[event].isna()
    mask &= (trials_df['choice'] != 0)
    return mask

# Combined with behavior validity
combined_mask = mask.values & wheel_valid & me_valid
```

iii. The agent matched the reference code's `load_trials_and_mask()` function parameters (min_rt=0.08, max_rt=2.0, max_trial_len=10.0, exclude no-choice). The additional behavior validity filtering matches the `align_spike_behavior()` function in the reference code which removes trials with missing behavior data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` and `spikes.clusters.npy` files from pykilosort spike sorting, along with `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` for brain region mapping.

ii.
```python
spike_times = np.load(os.path.join(ks_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(ks_dir, 'spikes.clusters.npy')).flatten()
clusters_channels = np.load(os.path.join(ks_dir, 'clusters.channels.npy')).flatten()
brain_ids = np.load(brain_id_files[0]).flatten()
```

iii. The agent identified spike times and cluster assignments as the raw neural data from the pykilosort sorting directory, consistent with the reference code's `load_spiking_data`.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 20ms time bins within each trial's window (stimOn - 0.5s to stimOn + 1.5s), producing spike counts per cluster per time bin. The result is stored as integer spike counts (uint8 dtype for storage efficiency). Multiple probes within a session are merged by concatenating clusters with offset cluster IDs.

ii.
```python
def bin_spikes_per_trial(spike_times, spike_clusters, n_clusters, trial_starts, trial_ends):
    binned_all = np.zeros((n_trials, n_clusters, N_BINS), dtype=np.float32)
    for trial_idx in valid_indices:
        t_beg = trial_starts[trial_idx]
        t_end = trial_ends[trial_idx]
        i_start = np.searchsorted(spike_times, t_beg, side='left')
        i_end = np.searchsorted(spike_times, t_end, side='left')
        times_curr = spike_times[i_start:i_end]
        clust_curr = spike_clusters[i_start:i_end]
        time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)
        time_bin_idx = np.clip(time_bin_idx, 0, N_BINS - 1)
        np.add.at(binned_all[trial_idx], (clust_curr, time_bin_idx), 1)
    return binned_all
```

iii. The agent implemented spike binning using searchsorted and np.add.at, which is functionally equivalent to the reference code's `bincount2D` approach. The bin size (20ms), window (-0.5, 1.5s), and number of bins (100) match the reference parameters.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No quality control filtering is applied to neurons/clusters. All clusters from pykilosort are included regardless of quality metrics (no QC filtering).

ii.
```python
# In load_spikes: no filtering on cluster quality
spike_times = np.load(os.path.join(ks_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(ks_dir, 'spikes.clusters.npy')).flatten()
# No QC filter applied
```

iii. The agent noted that the reference code uses `qc=None` in `load_spiking_data`, meaning all clusters are loaded without quality filtering. This matches the reference code's behavior in `0_data_caching.py`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). Each trial's spike data is extracted from the window `[stimOn - 0.5s, stimOn + 1.5s]` and binned into 100 time bins of 20ms each.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
stim_on = trials_df[ALIGN_TIME].values
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]
```

iii. The agent followed the task specification ("Temporally align based on stimulus onset") and the reference code's parameters (`align_time='stimOn_times'`, `time_window=(-0.5, 1.5)`).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20ms (0.02s), producing 100 time bins per trial. No additional rebinning is applied beyond the initial spike binning.

ii.
```python
BINSIZE = 0.02  # 20 ms
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
```

iii. The agent matched the reference code's bin size of 20ms and 100 time steps per trial, consistent with the Zhang2025 paper ("divided into 20-ms bins, producing T = 100 time steps").

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Time since stimulus onset is computed from the bin edges relative to the alignment event (stimOn_times). It is not derived from a specific raw data variable but rather constructed from the time window parameters.

ii.
```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
# Results in values from -0.49 to 1.49 (bin centers)
```

iii. The agent created a synthetic time vector representing the center of each 20ms bin relative to stimulus onset, spanning from -0.5s to 1.5s.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time vector is computed as bin centers: starting at TIME_WINDOW[0] + BINSIZE/2 = -0.49s, incrementing by BINSIZE = 0.02s, for N_BINS = 100 bins. The same vector is used for all trials.

ii.
```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
time_since_stim = time_since_stim.astype(np.float32)
```

iii. The agent computed bin centers rather than bin edges, which is a reasonable representation of time within each bin.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time vector is inherently aligned with the neural data since both use the same bin structure. Each time bin center corresponds to the same time bin in the neural data. The same time vector is replicated for every trial.

ii.
```python
input_list = []
for i in range(n_valid_trials):
    inp = np.vstack([
        time_since_stim[np.newaxis, :],  # (1, N_BINS) - same for all trials
        np.full((1, N_BINS), trial_num_in_block[i], dtype=np.float32)
    ])
    input_list.append(inp)
```

iii. Since neural data and time since stimulus onset share the same bin structure, alignment is implicit.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from `probabilityLeft` in the trials table. Block boundaries are detected when `probabilityLeft` changes value.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
trial_num_in_block = all_trial_nums[valid_idx]
```

iii. The agent used changes in `probabilityLeft` to detect block transitions, matching the IBL task structure where blocks have different prior probabilities.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A counter starts at 1 and increments for each trial. When `probabilityLeft` changes (indicating a new block), the counter resets to 1. The computation is done on ALL trials first (including filtered ones), then valid trial indices are selected.

ii.
```python
def compute_trial_number_in_block(prob_left):
    trial_nums = np.zeros(len(prob_left), dtype=np.float32)
    counter = 1
    for i in range(len(prob_left)):
        if i > 0 and prob_left[i] != prob_left[i-1]:
            counter = 1
        trial_nums[i] = counter
        counter += 1
    return trial_nums
```

iii. The agent correctly computed trial number in block on all trials before filtering, so that filtered trials don't create artificial block boundaries. The value is broadcast across all time bins for each trial.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column in the trials dataframe (`_ibl_trials.table.pqt`).

ii.
```python
choice = valid_trials['choice'].values.copy()
choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
```

iii. The agent identified choice from the trials table, consistent with the reference code.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The raw choice values (-1 for left, 1 for right) are mapped to binary categories: left = 0, right = 1. No-choice trials (choice == 0) are excluded by the trial mask. The per-trial value is broadcast across all time bins.

ii.
```python
choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
# In output:
np.full((1, N_BINS), choice_mapped[i], dtype=np.int64)
```

iii. The mapping follows the task specification ("left = 0, right = 1") and the reference code convention.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from `probabilityLeft` in the trials dataframe.

ii.
```python
prob_left = valid_trials['probabilityLeft'].values
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
```

iii. The agent identified `probabilityLeft` as the source variable, consistent with the reference code.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The continuous probability values are mapped to categorical integers: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Unknown values default to category 1 (0.5). The per-trial value is broadcast across all time bins.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
# In output:
np.full((1, N_BINS), prior_mapped[i], dtype=np.int64)
```

iii. The mapping follows the task specification ("0.2 -> 0, 0.5 -> 1, 0.8 -> 2").

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy` files.

ii.
```python
wheel_pos = np.load(pos_file).flatten()
wheel_ts = np.load(ts_file).flatten()
```

iii. The agent loaded raw wheel position and timestamps from numpy files in the session's alf directory.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Processing follows the reference code's wheel processing pipeline: (1) interpolate wheel position to 1000 Hz uniform sampling, (2) apply Butterworth lowpass filter (order 8, corner frequency 20 Hz), (3) compute velocity as the filtered derivative times sampling rate, (4) take absolute value to get speed.

ii.
```python
def load_wheel_speed(session_dir):
    t_interp = np.arange(wheel_ts[0], wheel_ts[-1], 1.0 / WHEEL_FS)
    pos_interp = interpolate.interp1d(wheel_ts, wheel_pos, kind='linear')(t_interp)
    sos = signal.butter(N=WHEEL_FILTER_ORDER, Wn=WHEEL_CORNER_FREQ / WHEEL_FS * 2,
                        btype='lowpass', output='sos')
    vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * WHEEL_FS
    speed = np.abs(vel)
    return t_interp, speed
```

iii. The agent replicated the brainbox `wheel.py` functions (`interpolate_position` and `velocity_filtered`) since the brainbox SessionLoader was unavailable.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 bins using quantile-based binning. The quantile boundaries (33rd and 67th percentiles) are computed across ALL valid timepoints across all trials in a session, and `np.digitize` maps values to categories 0, 1, 2.

ii.
```python
def discretize_time_varying(values, n_bins=N_DISC_BINS):
    flat = values[~np.isnan(values)].flatten()
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]  # [0.333, 0.667]
    boundaries = np.quantile(flat, quantiles)
    result = np.digitize(values, boundaries).astype(np.int64)
    return result

wheel_disc = discretize_time_varying(wheel_data, N_DISC_BINS)
```

iii. The agent chose quantile-based discretization to ensure approximately equal bin populations. The discretization is done per session (using all valid trials within the session).

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated to the same time bins as the neural data. The continuous wheel speed signal is interpolated to N_BINS=100 uniform time points within each trial's window [stimOn-0.5, stimOn+1.5]. The interpolation target points are bin centers.

ii.
```python
def interpolate_behavior_to_bins(beh_times, beh_values, trial_starts, trial_ends):
    x_interp = np.linspace(t_beg, t_end, N_BINS, endpoint=False) + BINSIZE / 2
    y_interp = interpolate.interp1d(
        beh_t_clean, beh_v_clean, kind='linear',
        fill_value='extrapolate')(x_interp)
    result[trial_idx] = y_interp
```

iii. The agent used linear interpolation to resample the continuous wheel speed signal to match the neural data's time bins, following the reference code's `get_behavior_per_interval` approach.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` (fallback), along with the corresponding camera timestamps (`_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`).

ii.
```python
def load_whisker_motion_energy(session_dir):
    me_file = find_file(session_dir, 'leftCamera.ROIMotionEnergy.npy')
    cam_file = find_file(session_dir, '_ibl_leftCamera.times.npy')
    if me_file and cam_file:
        me = np.load(me_file).flatten()
        cam_times = np.load(cam_file).flatten()
        if len(me) == len(cam_times) and len(me) > 0:
            return cam_times, me
    # Fallback to right camera
    me_file = find_file(session_dir, 'rightCamera.ROIMotionEnergy.npy')
    cam_file = find_file(session_dir, '_ibl_rightCamera.times.npy')
    # ...
```

iii. The agent followed the reference code which prefers left camera motion energy, consistent with the IBL convention.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are loaded directly (no additional processing beyond loading). They are then interpolated to the trial time bins using the same `interpolate_behavior_to_bins` function as wheel speed.

ii.
```python
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, trial_starts, trial_ends)
```

iii. The agent loaded pre-computed motion energy values and interpolated them to match the neural time bins.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Whisker motion energy is discretized into 3 bins using the same quantile-based approach as wheel speed. Quantile boundaries are computed across all valid timepoints in the session.

ii.
```python
me_disc = discretize_time_varying(me_data, N_DISC_BINS)
```

iii. Same quantile-based discretization as wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is interpolated to the same time bins as neural data using linear interpolation to bin centers within the trial window [stimOn-0.5, stimOn+1.5].

ii.
```python
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, trial_starts, trial_ends)
```

iii. Same interpolation approach as wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) Sessions missing spike data, trials, wheel, or motion energy are skipped entirely. (2) Trials with NaN in key fields are excluded by the trial mask. (3) Trials where behavior interpolation fails (insufficient data points, NaNs) are marked invalid. (4) A combined mask requires all data to be valid for a trial to be included. (5) Sessions with fewer than 2 valid trials are skipped. (6) If camera times and motion energy arrays have mismatched lengths, the session's ME data is treated as unavailable. (7) Unknown probabilityLeft values default to category 1 (0.5).

ii.
```python
# Behavior interpolation handles missing data:
if len(beh_v) < 2:
    valid[trial_idx] = False
    continue
nan_mask = ~np.isnan(beh_v)
if nan_mask.sum() < 2:
    valid[trial_idx] = False
    continue

# Combined mask
combined_mask = mask.values & wheel_valid & me_valid
if n_valid < 2:
    print(f"  Skipping {eid}: only {n_valid} valid trials")
    return None
```

iii. The agent implemented defensive handling that matches the reference code's `align_spike_behavior` function which removes trials with missing behavior data.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is spike binning (`bin_spikes_per_trial`), which iterates over all trials and uses searchsorted + np.add.at per trial. The conversion output shows binning takes 0.3-2.1 seconds per session. The full conversion processed 340 sessions, taking roughly 15-20 minutes total. Saving the large pickle file (20.7 GB) is also time-consuming.

ii.
```python
# Timing output from conversion:
# "Binning spikes (1239 clusters, 692 trials)... 1.6s"
# "Session processed in 3.2s"
```

iii. The agent printed timing information for the binning step as requested.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized: (1) The trial loop in `bin_spikes_per_trial` could potentially use a fully vectorized bincount2D approach. (2) The trial loop in `interpolate_behavior_to_bins` processes trials sequentially. (3) The `compute_trial_number_in_block` function uses a Python loop. (4) The input/output list construction loops could use numpy broadcasting.

ii.
```python
# Trial loop in bin_spikes_per_trial:
for trial_idx in valid_indices:
    # ... per-trial processing

# Trial loop in interpolate_behavior_to_bins:
for trial_idx in range(n_trials):
    # ... per-trial interpolation

# Python loop in compute_trial_number_in_block:
for i in range(len(prob_left)):
    if i > 0 and prob_left[i] != prob_left[i-1]:
        counter = 1
```

iii. The agent chose searchsorted-based spike selection as a vectorization, but the per-trial loop remains.

## 10-c. What processing does the code repeat multiple times?

i. (1) Spike binning is done for ALL trials before filtering, meaning spikes are binned for trials that will later be excluded. (2) Behavior interpolation is similarly done for all trials before the combined mask is applied. (3) The `find_file` function uses glob patterns for each file, which repeatedly scans directory listings.

ii.
```python
# Spikes binned for ALL trials, then filtered:
binned_spikes = bin_spikes_per_trial(spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)
# ... later:
neural_data = binned_spikes[valid_idx]  # Only valid trials used
```

iii. The agent bins ALL trials first and then applies the mask, which wastes computation on excluded trials. However, this approach simplifies the code logic.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Spike binning and behavior interpolation are performed for ALL trials (including those that fail quality filters), then only valid trials are kept. (2) The merged spike times are sorted by time after concatenation, but this sorting is not strictly necessary since searchsorted is used within trial-specific windows. (3) Wheel speed and whisker ME continuous values are computed and stored temporarily but only the discretized versions are saved in the output. (4) Neural data for 'root' and 'void' brain regions (which together comprise ~75,000 neurons) is included, though the reference paper notes these are often excluded from analysis.

ii.
```python
# Sorting merged spikes (potentially unnecessary):
sort_idx = np.argsort(merged_times, kind='stable')
merged_times = merged_times[sort_idx]
merged_clusters = merged_clusters[sort_idx]

# Continuous values computed but only discretized versions saved:
wheel_disc = discretize_time_varying(wheel_data, N_DISC_BINS)
me_disc = discretize_time_varying(me_data, N_DISC_BINS)
```

iii. The agent noted the file size concern (initially 84.7 GB) and converted to uint8 for efficiency, but the fundamental approach of processing all trials before filtering remains.
