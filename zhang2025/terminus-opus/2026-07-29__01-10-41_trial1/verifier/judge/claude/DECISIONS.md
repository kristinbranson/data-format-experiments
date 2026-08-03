# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads session metadata from `code/code_zhang2025/data/bwm_release.csv`, groups by `eid` to get per-session info with probe names, then iterates over sessions loading data files directly from the filesystem using glob patterns to find files in the ONE cache directory structure (`data/one_cache/{lab}/Subjects/{subject}/{date}/001/alf/`). It does NOT use the ONE API or `SessionLoader`/`SpikeSortingLoader`.

ii.
```python
bwm_df = pd.read_csv('code/code_zhang2025/data/bwm_release.csv', index_col=0)
session_groups = bwm_df.groupby('eid').agg({
    'lab': 'first', 'subject': 'first', 'date': 'first',
    'probe_name': list, 'pid': list
}).reset_index()
```

```python
spike_times = np.load(os.path.join(ks_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(ks_dir, 'spikes.clusters.npy')).flatten()
```

iii. The AI chose direct file loading rather than the ONE API, parsing the directory structure manually. The CONVERSION_NOTES document the file paths explored during Step 2 (Dataset Exploration), noting the ONE cache directory structure.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column of the `bwm_release.csv` file. As sessions are processed, subjects are tracked in a dictionary (`subject_map`) mapping subject name to an index. The `subject_idx` array records the index for each session.

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

iii. The subject identity comes from the CSV metadata. Subjects are added to the list in the order they are first encountered during iteration, rather than being sorted alphabetically as in the reference.

## 1-c. How are the data split into sessions?

i. Sessions are already individual rows in the `bwm_release.csv` (after grouping by `eid`). Each unique `eid` defines one session.

ii.
```python
session_groups = bwm_df.groupby('eid').agg({...}).reset_index()
for sess_idx, (_, row) in enumerate(session_groups.iterrows()):
    # process each session
```

iii. No splitting needed; sessions are already distinct in the metadata CSV.

## 1-d. How are the data split into trials?

i. Trials are loaded from the `_ibl_trials.table.pqt` parquet file, where each row is one trial.

ii.
```python
def load_trials(session_dir):
    trial_files = glob.glob(os.path.join(session_dir, 'alf', '*', '_ibl_trials.table.pqt'))
    trials_df = pd.read_parquet(trial_files[0])
    return trials_df
```

iii. The trials table is already one row per trial, so no additional splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies five filtering criteria: (1) reaction time between 0.08s and 2.0s, (2) trial length (feedback_times - goCue_times) <= 10.0s, (3) no NaN in key columns (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), (4) exclude no-choice trials (choice == 0), and (5) behavioral data availability (wheel and whisker motion energy must be interpolatable for the trial window). There is no explicit coverage check like the reference's `covered()` function.

ii.
```python
def create_trials_mask(trials_df):
    nan_exclude = ['stimOn_times', 'choice', 'feedback_times',
                   'probabilityLeft', 'firstMovement_times', 'feedbackType']
    mask = pd.Series(True, index=trials_df.index)
    rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
    mask &= (rt >= MIN_RT)
    mask &= (rt <= MAX_RT)
    trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
    mask &= (trial_len <= MAX_TRIAL_LEN)
    for event in nan_exclude:
        mask &= ~trials_df[event].isna()
    mask &= (trials_df['choice'] != 0)
    return mask
```

Additional behavioral validity check:
```python
combined_mask = mask.values & wheel_valid & me_valid
```

iii. The CONVERSION_NOTES state these filters come from the reference code's `load_trials_and_mask()` function (Step 1). The max_trial_len=10.0 filter and NaN exclusion list come from the Zhang2025 reference code rather than the human reference solution.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments). Brain region information comes from `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
spike_times = np.load(os.path.join(ks_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(ks_dir, 'spikes.clusters.npy')).flatten()
```

iii. These are the standard spike sorting outputs used by both the reference code and the human reference solution.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms time bins over a 2s window (-0.5 to 1.5s around stimulus onset), producing a (n_clusters, 100) spike count matrix per trial. Multiple probes are merged with cluster IDs offset. **Notably, the spike counts are NOT divided by the bin width to convert to firing rates** - the data remains as raw spike counts.

ii.
```python
BINSIZE = 0.02  # 20 ms
time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)
time_bin_idx = np.clip(time_bin_idx, 0, N_BINS - 1)
np.add.at(binned_all[trial_idx], (clust_curr, time_bin_idx), 1)
```

```python
neural_list = [neural_data[i].astype(np.float32) for i in range(n_valid_trials)]
```

iii. The CONVERSION_NOTES describe using 20ms bins matching the reference. However, the code does not divide counts by the bin width (0.02s) to produce firing rates in Hz, which the reference code does (`spike_counts(...) / BIN`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does NOT apply any quality control filtering to clusters/neurons. All clusters from the spike sorting are included regardless of their quality label. The CONVERSION_NOTES explicitly note "The code loads ALL clusters (not just good ones) - qc=None" based on the Zhang2025 reference code.

ii.
```python
def load_spikes(session_dir, probe_names):
    # ...
    spike_times = np.load(os.path.join(ks_dir, 'spikes.times.npy')).flatten()
    spike_clusters = np.load(os.path.join(ks_dir, 'spikes.clusters.npy')).flatten()
    # No quality filtering applied
    spike_clusters = spike_clusters + cluster_offset
    cluster_offset += n_clusters
```

iii. The AI's CONVERSION_NOTES (Step 1 and Step 4) explicitly note that the Zhang2025 reference code uses `qc=None`, meaning all clusters. The AI followed this approach rather than filtering by `label >= 1` as the human reference does. The CONVERSION_NOTES Step 4 states: "Code uses ALL clusters, not just good ones. Paper stats about well-isolated are for different analysis."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). Each trial's spikes are binned relative to stimulus onset in a window from -0.5s to 1.5s. The trial start time is `stimOn_times + (-0.5)` and trial end is `stimOn_times + 1.5`.

ii.
```python
stim_on = trials_df[ALIGN_TIME].values  # ALIGN_TIME = 'stimOn_times'
trial_starts = stim_on + TIME_WINDOW[0]  # TIME_WINDOW = (-0.5, 1.5)
trial_ends = stim_on + TIME_WINDOW[1]
```

```python
time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)
```

iii. Both the instructions and reference code specify alignment to stimulus onset with window (-0.5, 1.5)s.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20ms bins, producing 100 time steps per trial over the 2s window. No rebinning is applied; spikes are binned directly into 20ms bins.

ii.
```python
BINSIZE = 0.02  # 20 ms
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100
```

iii. Matches the reference code's `binsize: 0.02` and the paper's "20-ms bins, producing T = 100 time steps."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from the binning grid itself, computed as the bin centers of the 100 time bins spanning -0.5 to 1.5s around stimulus onset.

ii.
```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
```

iii. This is a deterministic variable defined by the time window and bin size, not derived from raw experimental data.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. Simple arithmetic: compute the center of each 20ms bin in the window [-0.5, 1.5]s. The same values are used for every trial.

ii.
```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
```

iii. No complex processing required; this matches the reference approach.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time values are the centers of the same bins used for the neural data, so they are inherently aligned. Both share the same 100-bin grid.

ii.
```python
# Neural bins:
time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)
# Input values (same grid centers):
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
```

iii. Alignment is guaranteed by construction since both use the same binning grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `probabilityLeft` in the trials table. Block boundaries are detected where `probabilityLeft` changes value.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
```

iii. The trials table does not contain an explicit block identifier, so blocks must be inferred from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial number counts the position within each block, starting from 1 (not 0). Block boundaries are detected by comparing consecutive `probabilityLeft` values. The count is computed on ALL trials before filtering, so dropped trials still advance the counter, preserving the animal's real position in the block.

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

iii. The AI computes trial numbers before filtering and then selects valid trials' values, matching the reference's approach of computing `trial_variables` before filtering. However, the AI starts counting from 1 while the reference uses `cumcount()` which starts from 0.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, which takes values +1 (left), -1 (right), and 0 (no response). No-response trials are excluded.

ii.
```python
choice = valid_trials['choice'].values.copy()
choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
```

iii. The choice variable is directly available in the trials table.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps choice values as: -1 -> 0 (labeled "left" in the comment) and 1 -> 1 (labeled "right"). **However, the comment is incorrect about the IBL convention**: in IBL, choice=+1 means LEFT and choice=-1 means RIGHT. So the AI's mapping actually assigns RIGHT=0 and LEFT=1, which is the opposite of the instructions ("left = 0, right = 1") and the reference code.

ii.
```python
# Choice: -1 -> 0 (left), 1 -> 1 (right)  # <-- Comment is wrong about IBL convention
choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
```

Reference code for comparison:
```python
CHOICE = {1.0: 0, -1.0: 1}  # +1 is a leftward choice, -1 rightward
```

iii. The AI misidentified which IBL choice value corresponds to left vs right, leading to a reversed mapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
prob_left = valid_trials['probabilityLeft'].values
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
```

iii. The mapping follows the instructions: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Simple lookup mapping. Values not in {0.2, 0.5, 0.8} default to 1 (the 0.5 category) via `dict.get(p, 1)`.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
```

iii. The reference instead filters out trials where `probabilityLeft` is not in {0.2, 0.5, 0.8} via `trials.probabilityLeft.isin(PRIOR)`, so unknown values never reach the mapping step. The AI's approach of defaulting to 1 could silently include invalid values.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`. The AI reimplements the wheel processing pipeline (interpolation, filtering, velocity computation) rather than using `SessionLoader`.

ii.
```python
wheel_pos = np.load(pos_file).flatten()
wheel_ts = np.load(ts_file).flatten()
```

iii. These are the same raw wheel files used by the reference, just loaded directly rather than through the ONE API.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps: (1) Interpolate wheel position to 1000 Hz uniform sampling, (2) Apply 8th-order 20 Hz Butterworth lowpass filter and differentiate to get velocity, (3) Take absolute value to get speed. The continuous speed is then interpolated to the 100 bin centers per trial, and discretized into 3 categories.

ii.
```python
pos_interp = interpolate.interp1d(wheel_ts, wheel_pos, kind='linear')(t_interp)
sos = signal.butter(N=WHEEL_FILTER_ORDER, Wn=WHEEL_CORNER_FREQ / WHEEL_FS * 2,
                    btype='lowpass', output='sos')
vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * WHEEL_FS
speed = np.abs(vel)
```

iii. This reimplements the same pipeline as `brainbox.behavior.wheel.velocity_filtered`, matching the reference's approach.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Discretized into 3 bins using quantile-based boundaries. The boundaries are computed as the 1/3 and 2/3 quantiles of all wheel speed values across all valid trials in the session.

ii.
```python
def discretize_time_varying(values, n_bins=N_DISC_BINS):
    flat = values[~np.isnan(values)].flatten()
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]  # [0.333, 0.667]
    boundaries = np.quantile(flat, quantiles)
    result = np.digitize(values, boundaries).astype(np.int64)
    return result
```

iii. This matches the reference's approach: `np.digitize(trace, np.percentile(trace, [100/3, 200/3]))`. Both compute session-level percentiles to create roughly equal-sized categories.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed trace is interpolated to the same 100 bin centers used for the neural data, ensuring bin-for-bin alignment.

ii.
```python
x_interp = np.linspace(t_beg, t_end, N_BINS, endpoint=False) + BINSIZE / 2
y_interp = interpolate.interp1d(beh_t_clean, beh_v_clean, kind='linear',
                                 fill_value='extrapolate')(x_interp)
```

iii. The AI uses `scipy.interpolate.interp1d` with `fill_value='extrapolate'` while the reference uses `np.interp`. The AI also uses `np.linspace` to compute bin centers while the reference uses `onset + TIME`. Both target the same 100 time points.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy` (fallback), together with the corresponding camera timestamps (`_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`).

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
    # Try right camera as fallback...
```

iii. Matches the reference's camera preference (left first, then right).

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy trace is used as-is (no filtering or normalization). It is interpolated to the 100 bin centers per trial, then discretized into 3 bins using session-level quantiles, same as wheel speed.

ii.
```python
me_binned, me_valid = interpolate_behavior_to_bins(me_times, me_values, trial_starts, trial_ends)
me_disc = discretize_time_varying(me_data, N_DISC_BINS)
```

iii. Matches the reference approach of using the released trace without additional processing.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same method as wheel speed: discretized into 3 bins using the 33rd and 67th percentile of all whisker motion energy values across valid trials in the session.

ii.
```python
me_disc = discretize_time_varying(me_data, N_DISC_BINS)
```

iii. Same approach as the reference.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated to the same 100 bin centers as the neural data.

ii.
```python
me_binned, me_valid = interpolate_behavior_to_bins(me_times, me_values, trial_starts, trial_ends)
```

iii. Same alignment method as wheel speed and neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple levels of handling: (1) Sessions without trials data, spike data, or with fewer than 2 valid trials are skipped entirely. (2) Trials with NaN in key fields are filtered out. (3) Trials where wheel or whisker motion energy cannot be interpolated (fewer than 2 valid data points in the window) are marked invalid. (4) Exceptions during session processing are caught and the session is skipped. (5) Missing probe data within a session is skipped.

ii.
```python
if trials_df is None:
    return None
if spike_times is None:
    return None
combined_mask = mask.values & wheel_valid & me_valid
if n_valid < 2:
    return None
```

iii. The AI handles missing data by skipping at the trial or session level, which is reasonable. The reference uses coverage checks (`covered()`) rather than interpolation failure to determine trial validity.

## 10-a. What are the most time-consuming steps of the code?

i. The AI's code processes sessions sequentially (no parallelization). The most time-consuming steps are: (1) loading spike data from disk (large numpy files), (2) binning spikes per trial, and (3) interpolating behavioral signals per trial.

ii.
```python
spike_times = np.load(os.path.join(ks_dir, 'spikes.times.npy')).flatten()
# ...
binned_spikes = bin_spikes_per_trial(spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)
```

iii. The AI prints timing information for the binning step. The reference uses `ProcessPoolExecutor` with 10 workers for parallel processing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops iterate per-trial: (1) spike binning (`bin_spikes_per_trial`) loops over trials, (2) behavior interpolation (`interpolate_behavior_to_bins`) loops over trials. Both could potentially be vectorized.

ii.
```python
for trial_idx in valid_indices:
    # spike binning per trial
    np.add.at(binned_all[trial_idx], (clust_curr, time_bin_idx), 1)
```

```python
for trial_idx in range(n_trials):
    # behavior interpolation per trial
    y_interp = interpolate.interp1d(...)(x_interp)
```

iii. The per-trial loop for spike binning is similar to the reference's approach. The behavior interpolation loop could be partially vectorized but each trial has different time ranges.

## 10-c. What processing does the code repeat multiple times?

i. The AI bins all spikes for ALL trials before filtering (applying the trial mask), meaning spikes are binned for trials that are subsequently discarded. The reference computes trial filtering first and only bins spikes for valid trials.

ii.
```python
# Bins spikes for ALL trials
binned_spikes = bin_spikes_per_trial(spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)
# Then filters
neural_data = binned_spikes[valid_idx]
```

iii. This is wasteful because invalid trials' spike data is computed then discarded.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Two main sources of unnecessary processing: (1) Spike binning is performed for ALL clusters (no QC filtering), meaning spikes from low-quality clusters are binned but ideally should have been filtered out. (2) As noted above, spikes are binned for all trials before filtering, wasting computation on trials that are later discarded.

ii.
```python
# All clusters included, no QC filter
n_clusters = len(cluster_regions)
binned_spikes = bin_spikes_per_trial(spike_times, spike_clusters, n_clusters, trial_starts, trial_ends)
```

iii. The lack of QC filtering means the neural data includes many low-quality units that add noise without useful signal.
