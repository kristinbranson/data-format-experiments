# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads a CSV file (`bwm_release.csv`) from the reference code directory to get session metadata (eid, subject, date, lab, probe names). It then walks the filesystem under `data/one_cache/` to build a mapping from (subject, date) to session directory paths. Data files (trials, spikes, wheel, whisker) are loaded directly from the filesystem using numpy and pandas, not through the ONE API.

ii.
```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
session_map = build_session_map(DATA_DIR)
# ...
for eid, group in bwm_df.groupby('eid'):
    subject = group['subject'].iloc[0]
    date = group['date'].iloc[0]
    key = (subject, date)
    if key in session_map:
        sessions_info.append({...})
```

iii. The AI chose to use the bwm_release.csv to identify sessions and then resolve the data files on disk directly, rather than using the ONE API. The CONVERSION_NOTES mention "Matched to bwm_release: 459/459" indicating the CSV was used as the session index.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column of the bwm_release.csv. A dictionary `subject_to_idx` maps each unique subject name to an integer index, built incrementally as sessions are processed.

ii.
```python
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(all_subjects)
    all_subjects.append(subject)
all_subject_idx.append(subject_to_idx[subject])
```

iii. The subject names come directly from the CSV metadata. The ordering depends on the iteration order through sessions rather than alphabetical sorting.

## 1-c. How are the data split into sessions?

i. Each row group in the bwm_release.csv (grouped by eid) represents one session. Each session is processed independently via `process_session()`.

ii.
```python
for eid, group in bwm_df.groupby('eid'):
    # ...
    sessions_info.append({'eid': eid, 'subject': subject, ...})
```

iii. Sessions are defined by unique eid values in the CSV.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) has one row per trial. After loading, a mask is applied to filter valid trials.

ii.
```python
trials_df = pd.read_parquet(trials_file)
mask = create_trials_mask(trials_df)
valid_trials_df = trials_df[mask].reset_index(drop=True)
```

iii. The trials table already has one row per trial, so no splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. Multiple criteria are applied: (1) reaction time must be between 80ms and 2s, (2) trial duration must be <= 10s (feedback_times - goCue_times), (3) NaN exclusion in stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType, (4) no-choice trials (choice==0) excluded, (5) additionally, trials must have valid wheel and whisker data in the time window.

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']
# ...
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
```

iii. The AI based its trial filtering on the `load_trials_and_mask` function from the reference code, noting defaults for NaN exclusion and max_trial_len=10.0. The reaction time bounds (0.08-2.0s) were also applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments). Cluster-to-brain-region mapping uses `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(version_dir, 'spikes.clusters.npy')).flatten()
cluster_channels = np.load(os.path.join(version_dir, 'clusters.channels.npy')).flatten()
channel_brain_ids = np.load(os.path.join(version_dir, 'channels.brainLocationIds_ccf_2017.npy')).flatten()
```

iii. The AI loads the same spike sorting files as the reference.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms bins over the 2s trial window (-0.5 to 1.5s relative to stimulus onset), giving spike COUNTS per cluster per bin. Multiple probes are merged by offsetting cluster IDs. The neural data is stored as raw spike counts (not converted to firing rates in Hz).

ii.
```python
bin_idx = np.minimum(
    ((trial_times - t_beg) / binsize).astype(np.int32),
    n_bins - 1
)
linear_idx = trial_clusters * n_bins + bin_idx
np.add.at(binned[trial_idx].ravel(), linear_idx, 1)
# ...
neural_list.append(final_spikes[t].astype(np.float32))  # raw counts, not divided by BIN
```

iii. The AI documents the binning approach in CONVERSION_NOTES but does not mention converting to firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No quality control filtering is applied to clusters. All clusters are used regardless of their quality label.

ii.
```python
# No QC filtering visible in load_spikes() - all clusters from pykilosort are used
spike_clusters = np.load(os.path.join(version_dir, 'spikes.clusters.npy')).flatten()
# No filtering by label or metrics
```

iii. The AI explicitly decided not to filter by QC, noting in CONVERSION_NOTES: "No QC filtering: `load_spiking_data` called without qc parameter (defaults to None = all clusters)". This is based on the reference code's `prepare_data` function which calls `load_spiking_data` with default qc=None.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spike window is defined by `stim_on + TIME_WINDOW[0]` to `stim_on + TIME_WINDOW[1]`, i.e., -0.5s to 1.5s relative to stimulus onset. Spikes in this window are binned relative to the window start.

ii.
```python
stim_on = valid_trials_df[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
# ...
bin_idx = np.minimum(
    ((trial_times - t_beg) / binsize).astype(np.int32),
    n_bins - 1
)
```

iii. The alignment to stimulus onset matches the reference code parameters.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20ms bins (BINSIZE = 0.02), producing 100 time bins over the 2s window. No rebinning is applied.

ii.
```python
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 time bins
```

iii. Matches the reference code's binsize of 0.02s.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not derived from any raw data variable. It is computed as the bin centers of the time grid from -0.5s to 1.5s.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
).astype(np.float32)
```

iii. The time input is derived from the binning parameters, not from any data variable.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The bin centers are computed using `np.linspace` from -0.49 to 1.49, producing 100 evenly-spaced values.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
).astype(np.float32)
```

iii. No processing of raw data; this is a deterministic time axis.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time values represent bin centers of the same 20ms bins used for spike binning, so they are aligned by construction.

ii.
```python
# Bin centers match the neural binning grid
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
).astype(np.float32)
```

iii. The time axis is the same grid as the neural data bins.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A change in probabilityLeft indicates a new block.

ii.
```python
def compute_trial_number_in_block(prob_left):
    for i in range(len(prob_left)):
        if prob_left[i] != current_prob:
            current_block_start = i
            current_prob = prob_left[i]
        trial_nums[i] = i - current_block_start + 1
```

iii. Blocks are inferred from changes in probabilityLeft, as no explicit block identifier exists in the trials table.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial number within each block is computed by detecting when `probabilityLeft` changes value, then counting from 1 within each block. The count is computed on ALL trials before filtering, so filtered-out trials still advance the counter. The result starts at 1, not 0.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
valid_indices = np.where(mask.values)[0]
trial_nums_valid = all_trial_nums[valid_indices]
trial_nums_final = trial_nums_valid[combined_valid]
```

iii. The AI computes block numbers on unfiltered trials to preserve the true position, then selects the values for valid trials.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column in the trials table, which has values -1, 0, and 1.

ii.
```python
choice = final_trials['choice'].values.copy()
choice[choice == -1] = 0
choice = choice.astype(np.float32)
```

iii. The AI maps choice=-1 to 0 and choice=1 stays as 1.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI remaps choice values: -1 maps to 0 and +1 stays as 1. No-choice trials (choice==0) were already filtered out. However, in IBL conventions, choice=+1 is LEFT and choice=-1 is RIGHT. The instructions specify left=0, right=1. So the AI's mapping gives LEFT=1 and RIGHT=0, which is inverted from the instructions.

ii.
```python
# AI maps -1->0, +1->1
# But IBL: +1=left, -1=right
# Instructions: left=0, right=1
# Correct mapping should be: +1->0, -1->1
choice[choice == -1] = 0  # right -> 0 (should be 1)
# choice=1 (left) stays as 1 (should be 0)
```

iii. The AI's CONVERSION_NOTES state "Remap: -1->0 (left), 1->1 (right)" indicating the AI believed -1 meant left and +1 meant right, which is the opposite of IBL convention.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
prob_left = final_trials['probabilityLeft'].values.copy()
prior = np.zeros(len(prob_left), dtype=np.float32)
prior[np.isclose(prob_left, 0.2)] = 0
prior[np.isclose(prob_left, 0.5)] = 1
prior[np.isclose(prob_left, 0.8)] = 2
```

iii. The mapping follows the decoder task specification: 0.2->0, 0.5->1, 0.8->2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct mapping of 0.2->0, 0.5->1, 0.8->2 using `np.isclose` for floating point comparison.

ii.
```python
prior[np.isclose(prob_left, 0.2)] = 0
prior[np.isclose(prob_left, 0.5)] = 1
prior[np.isclose(prob_left, 0.8)] = 2
```

iii. No additional processing beyond the mapping.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The raw wheel position and timestamps are loaded directly from disk.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel velocity is computed using a Butterworth-filtered derivative (`velocity_filtered` from brainbox). The speed is the absolute value. However, the AI does NOT first interpolate the position to a uniform 1000Hz grid as `SessionLoader` does -- it computes the sampling frequency from the median of `np.diff(timestamps)` and passes the raw position directly to `velocity_filtered`. The speed is then interpolated onto trial time bins and discretized into 3 equal-frequency bins using GLOBAL percentiles across all sessions.

ii.
```python
dt_median = np.median(np.diff(timestamps))
fs = 1.0 / dt_median
velocity, _ = velocity_filtered(position, fs)
speed = np.abs(velocity)
# ...
# Interpolation to trial bins:
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
interp_func = interp1d(local_times, local_vals, kind='linear', fill_value='extrapolate')
# ...
# Global discretization:
wheel_percentiles = np.percentile(all_wheel_vals, [33.33, 66.67])
wheel_disc = np.digitize(wheel_raw, wheel_percentiles)
```

iii. The AI used the reference code's `velocity_filtered` function but skipped the interpolation to 1000Hz that SessionLoader performs before calling velocity_filtered. The discretization uses global percentiles rather than per-session percentiles.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Three equal-frequency bins using GLOBAL percentiles at 33.33% and 66.67% across all sessions combined, producing categories 0 (low), 1 (medium), 2 (high).

ii.
```python
all_wheel_vals = np.concatenate([w.flatten() for w in all_wheel_raw])
wheel_percentiles = np.percentile(all_wheel_vals[~np.isnan(all_wheel_vals)], [33.33, 66.67])
wheel_disc = np.digitize(wheel_raw, wheel_percentiles).astype(np.int64)
```

iii. The AI chose global discretization rather than per-session.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated onto trial time points defined by `np.linspace(t_beg + binsize, t_end, n_bins)` where t_beg = stim_on + TIME_WINDOW[0] and t_end = stim_on + TIME_WINDOW[1]. This grid differs slightly from bin centers: the first point is at t_beg + 0.02 = stim_on - 0.48 rather than stim_on - 0.49 (the true bin center), and the last point is at t_end = stim_on + 1.5 rather than stim_on + 1.49.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. The AI's CONVERSION_NOTES reference the code's interpolation approach: "x_interp = np.linspace(interval_beg + binsize, interval_end, n_bins)".

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (or `rightCamera.ROIMotionEnergy.npy` as fallback) and the corresponding `_ibl_<side>Camera.times.npy`.

ii.
```python
me_file = find_file(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_dir, '_ibl_leftCamera.times.npy')
if me_file is None or times_file is None:
    me_file = find_file(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_file(alf_dir, '_ibl_rightCamera.times.npy')
```

iii. Left camera preferred, right as fallback, matching the reference approach.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The motion energy values are loaded as-is, NaN values are removed, and the trace is interpolated onto trial time bins using the same grid as wheel speed. Then discretized into 3 global equal-frequency bins.

ii.
```python
me_values = np.load(me_file).flatten()
me_times = np.load(times_file).flatten()
valid = ~(np.isnan(me_values) | np.isnan(me_times))
me_values = me_values[valid]
me_times = me_times[valid]
# then interpolated and discretized same as wheel
```

iii. No additional smoothing or normalization is applied.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Three equal-frequency bins using GLOBAL percentiles at 33.33% and 66.67% across all sessions.

ii.
```python
all_me_vals = np.concatenate([m.flatten() for m in all_me_raw])
me_percentiles = np.percentile(all_me_vals[~np.isnan(all_me_vals)], [33.33, 66.67])
me_disc = np.digitize(me_raw, me_percentiles).astype(np.int64)
```

iii. Same global discretization approach as wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same interpolation approach as wheel speed: `np.linspace(t_beg + binsize, t_end, n_bins)`, which has a slight offset from the true bin centers used for neural data.

ii.
```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
```

iii. Same interpolation grid as wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) NaN values in key trial columns cause trial exclusion, (2) Sessions with no spike data or <2 valid trials are skipped, (3) Whisker ME length mismatches are handled by truncating to minimum length, (4) NaN values in whisker ME are removed, (5) Trials without valid wheel or whisker interpolation are excluded via combined_valid mask, (6) Sessions missing whisker ME entirely are skipped (15 sessions).

ii.
```python
# Length mismatch handling
min_len = min(len(me_values), len(me_times))
me_values = me_values[:min_len]

# NaN removal
valid = ~(np.isnan(me_values) | np.isnan(me_times))

# Combined validity
combined_valid = wheel_valid & me_valid
if n_final < 2:
    return None
```

iii. The AI handles missing data by exclusion at various levels (trial, session).

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike data from disk (reading large .npy files) and the spike binning loop over trials.

ii.
```python
spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
# ...
for trial_idx in range(n_trials):
    # spike binning per trial
```

iii. The conversion took ~37 minutes for 459 sessions.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates per trial, and the behavior interpolation also loops per trial. Both could potentially be vectorized. The spike binning in `bin_spikes_fast` still loops over trials with a per-trial searchsorted + add.at pattern. The behavior interpolation in `interpolate_behavior_to_bins` uses scipy's interp1d per trial.

ii.
```python
for trial_idx in range(n_trials):
    # ...
    np.add.at(binned[trial_idx].ravel(), linear_idx, 1)

for trial_idx in range(n_trials):
    # ...
    interp_func = interp1d(local_times, local_vals, ...)
```

iii. The per-trial loops are the main inefficiency.

## 10-c. What processing does the code repeat multiple times?

i. The code has two spike binning functions (`bin_spikes_vectorized` and `bin_spikes_fast`) -- only `bin_spikes_fast` is used. There is no repeated processing of the same data within a session.

ii.
```python
def bin_spikes_vectorized(...):  # unused
def bin_spikes_fast(...):        # used
```

iii. The dead code (`bin_spikes_vectorized`) is unused but present.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `cluster_channels` and `channel_brain_ids` for brain region mapping, which involves additional file I/O. It also constructs `interp1d` objects per trial which is heavier than `np.interp`. The dead `bin_spikes_vectorized` function is compiled but never called.

ii.
```python
cluster_channels = np.load(os.path.join(version_dir, 'clusters.channels.npy')).flatten()
channel_brain_ids = np.load(os.path.join(version_dir, 'channels.brainLocationIds_ccf_2017.npy')).flatten()
```

iii. Brain region mapping is needed for the output format, so the file I/O is not strictly unnecessary.
