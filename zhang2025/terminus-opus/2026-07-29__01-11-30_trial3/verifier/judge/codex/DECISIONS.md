# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the ONE API. It reads `code/code_zhang2025/data/bwm_release.csv` to get released session metadata, walks the local `data/one_cache` directory tree to map `(subject, date)` to a session path, then opens parquet and `.npy` files directly from each session's `alf/` directory.

ii. ```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
session_map = build_session_map(DATA_DIR)

for eid, group in bwm_df.groupby('eid'):
    subject = group['subject'].iloc[0]
    date = group['date'].iloc[0]
    probe_names = list(group['probe_name'].unique())
    key = (subject, date)
    if key in session_map:
        sessions_info.append({
            'eid': eid,
            'subject': subject,
            'probe_names': probe_names,
            'path': session_map[key]
        })
```

```python
trials_df = pd.read_parquet(trials_file)
spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(version_dir, 'spikes.clusters.npy')).flatten()
```

iii. The notes say the plan was to 'Load all sessions from bwm_release.csv' and use the cache directly. The justification in the trajectory was pragmatic: all 459 released sessions could be matched locally from the release CSV and filesystem without going through ONE.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `subject` column of `bwm_release.csv`. During assembly, the script assigns a new subject index the first time a subject is encountered while iterating sessions; it does not pre-group sessions by subject or sort the subject list.

ii. ```python
for eid, group in bwm_df.groupby('eid'):
    subject = group['subject'].iloc[0]
```

```python
subject_to_idx = {}
...
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(all_subjects)
    all_subjects.append(subject)
...
all_subject_idx.append(subject_to_idx[subject])
```

iii. The notes repeatedly treat the CSV's `subject` field as authoritative session metadata. There is no separate justification beyond reusing release-table metadata.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in `bwm_release.csv`. For each `eid`, the script collects the listed probes, finds one cache directory via `(subject, date)`, and processes that as a session.

ii. ```python
for eid, group in bwm_df.groupby('eid'):
    subject = group['subject'].iloc[0]
    date = group['date'].iloc[0]
    probe_names = list(group['probe_name'].unique())
    key = (subject, date)
    if key in session_map:
        sessions_info.append({
            'eid': eid,
            'subject': subject,
            'date': date,
            'probe_names': probe_names,
            'path': session_map[key]
        })
```

iii. The notes say 'Use 459 from bwm_df' and use all probes listed for each `eid`.

## 1-d. How are the data split into trials?

i. Trials come from rows of `_ibl_trials.table.pqt`. The script loads the parquet table, applies a Boolean mask, then resets the filtered table index so each remaining row becomes one trial in the converted session.

ii. ```python
def load_trials(alf_dir):
    trials_file = find_file(alf_dir, '_ibl_trials.table.pqt')
    ...
    trials_df = pd.read_parquet(trials_file)
    return trials_df
```

```python
mask = create_trials_mask(trials_df)
valid_trials_df = trials_df[mask].reset_index(drop=True)
```

iii. The agent's notes describe the trials table as the row-wise source of trial structure, matching the way the reference code is organized.

## 1-e. How are trials filtered based on quality controls?

i. The script first filters trials with a custom mask: reaction time must be between 80 ms and 2 s, trial duration must be at most 10 s, several trial columns must be non-NaN, unbiased trials are kept, and no-choice trials are removed. After interpolating wheel and whisker traces, it applies a second mask keeping only trials where both interpolations succeeded. If fewer than two trials remain, the session is dropped.

ii. ```python
MIN_RT = 0.08
MAX_RT = 2.0
MAX_TRIAL_LEN = 10.0
EXCLUDE_UNBIASED = False
EXCLUDE_NOCHOICE = True
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']
```

```python
def create_trials_mask(trials_df):
    query_parts = []
    query_parts.append(f'(firstMovement_times - stimOn_times < {MIN_RT})')
    query_parts.append(f'(firstMovement_times - stimOn_times > {MAX_RT})')
    query_parts.append(f'(feedback_times - goCue_times > {MAX_TRIAL_LEN})')
    for event in NAN_EXCLUDE:
        query_parts.append(f'{event}.isnull()')
    if EXCLUDE_NOCHOICE:
        query_parts.append('(choice == 0)')
    query = ' | '.join(query_parts)
    mask = ~trials_df.eval(query)
    return mask
```

```python
wheel_binned, wheel_valid = interpolate_behavior_to_bins(...)
me_binned, me_valid = interpolate_behavior_to_bins(...)
combined_valid = wheel_valid & me_valid
```

iii. In Step 58 of the trajectory, the AI explicitly justified keeping unbiased trials because prior decoding needed a `0.5` class. The notes also claim this matches the reference `load_trials_and_mask` defaults, and later justify skipped sessions as a consequence of missing whisker motion energy.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted neural arrays are built from `spikes.times.npy` and `spikes.clusters.npy` for each probe. The script also reads `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`, but those are used to assign brain regions rather than generate the neural time series itself.

ii. ```python
spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(version_dir, 'spikes.clusters.npy')).flatten()
cluster_channels = np.load(os.path.join(version_dir, 'clusters.channels.npy')).flatten()
channel_brain_ids = np.load(os.path.join(version_dir, 'channels.brainLocationIds_ccf_2017.npy')).flatten()
```

iii. The notes and trajectory consistently identify the spike times and cluster assignments as the neural source variables, with region arrays treated separately for metadata.

## 2-b. How is the `neural` data processed?

i. For each session, all listed probes are merged by offsetting cluster IDs so they are unique across probes, concatenating spikes, and sorting by time. Each trial window runs from `stimOn_times - 0.5` to `stimOn_times + 1.5`, and spikes are counted into 100 bins of width 20 ms. The result is kept as spike counts in `float32`; the script does not divide by bin width to convert counts to firing rates.

ii. ```python
all_spike_clusters.append(spike_clusters + cluster_offset)
...
merged_times = np.concatenate(all_spike_times)
merged_clusters = np.concatenate(all_spike_clusters)
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

```python
neural_list.append(final_spikes[t].astype(np.float32))
```

iii. The notes say the agent intended to 'Bin spikes into 20ms bins aligned to stimOn_times'. It also believed all clusters should be used, so no rate conversion or QC-related postprocessing was added around the binned counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered by unit quality. The script loads every cluster present in the selected probe directories and keeps them all; there is no use of cluster QC labels or spike-sorting metrics.

ii. ```python
def load_spikes(alf_dir, probe_name):
    ...
    spike_times = np.load(...)
    spike_clusters = np.load(...)
    cluster_channels = np.load(...)
    channel_brain_ids = np.load(...)
    return spike_times, spike_clusters, cluster_channels, channel_brain_ids
```

```python
for probe_name in probe_names:
    result = load_spikes(alf_dir, probe_name)
    if result[0] is not None:
        probes_data.append(result)
```

iii. The notes repeatedly justify this as 'No QC filtering' because the agent believed the reference code used `qc=None` and therefore kept all clusters, including `root`, `void`, and `x` regions (Trajectory Step 189).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to `stimOn_times`. For each kept trial the binning interval starts at `stimOn_times - 0.5` and ends at `stimOn_times + 1.5`; spikes are placed into bins by subtracting the interval start `t_beg`, which is equivalent to expressing them relative to stimulus onset over that fixed window.

ii. ```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

```python
stim_on = valid_trials_df[ALIGN_TIME].values
interval_begs = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

```python
bin_idx = np.minimum(
    ((trial_times - t_beg) / binsize).astype(np.int32),
    n_bins - 1
)
```

iii. The notes and trajectory consistently say the decoder should be aligned to stimulus onset with a `(-0.5, 1.5)` s window, mirroring the method paper.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20 ms per bin, with 100 bins across a 2 s window. There is no rebinning to a different coarser resolution after trial construction.

ii. ```python
BINSIZE = 0.02  # 20ms bins
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]  # 2.0 seconds
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 time bins
```

iii. The notes explicitly cite the method paper's '20ms bins' and 'T=100' as the intended match.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is indirectly anchored to the trial `stimOn_times` variable via the global alignment settings. The actual stored values are synthetic bin-center timestamps for the fixed `(-0.5, 1.5)` window rather than a raw field copied from disk.

ii. ```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
```

```python
stim_on = valid_trials_df[ALIGN_TIME].values
```

```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
).astype(np.float32)
```

iii. The notes describe this input as a computed time axis based on the chosen alignment event and binning scheme, not as a direct raw-data column.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The script creates a fixed `float32` vector of bin centers from `-0.49` to `1.49` seconds using `np.linspace`, then reuses that same vector for every trial.

ii. ```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
).astype(np.float32)
```

iii. The notes justify this as simply encoding the chosen decoding grid.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the same number of bins and the same bin-center convention as the neural trial matrices, so it is aligned bin-for-bin with the spike-count arrays.

ii. ```python
# Neural bins are defined from the same TIME_WINDOW and BINSIZE
interval_begs = stim_on + TIME_WINDOW[0]
...
# Input time axis uses the corresponding bin centers
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE/2,
    TIME_WINDOW[1] - BINSIZE/2,
    N_BINS
).astype(np.float32)
```

iii. The notes describe all decoder inputs as aligned to the same stimulus-locked 20 ms grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial table's `probabilityLeft` column. A new block is declared whenever `probabilityLeft` changes value.

ii. ```python
def compute_trial_number_in_block(prob_left):
    ...
    current_prob = prob_left[0]
    for i in range(len(prob_left)):
        if prob_left[i] != current_prob:
            current_block_start = i
            current_prob = prob_left[i]
```

iii. The notes explicitly describe block structure as being recovered from `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The script scans all trials in the session, resets the counter when `probabilityLeft` changes, and assigns `i - current_block_start + 1`. Because it computes the counts on the full unfiltered trial table and only masks afterwards, filtered-out trials still advance the within-block position. The numbering is 1-based, not 0-based.

ii. ```python
trial_nums = np.zeros(len(prob_left), dtype=np.float32)
current_block_start = 0
current_prob = prob_left[0]
for i in range(len(prob_left)):
    if prob_left[i] != current_prob:
        current_block_start = i
        current_prob = prob_left[i]
    trial_nums[i] = i - current_block_start + 1
```

```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
valid_indices = np.where(mask.values)[0]
trial_nums_valid = all_trial_nums[valid_indices]
trial_nums_final = trial_nums_valid[combined_valid]
```

iii. The notes say trial number should be computed from all trials 'so the block count is real', and later note that it 'starts at 1'.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is taken from the trial table's `choice` column after trial filtering.

ii. ```python
choice = final_trials['choice'].values.copy()
```

iii. The notes and trajectory consistently identify `trials_df.choice` as the source variable.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The script removes no-choice trials earlier in the mask, then recodes `-1` to `0` and leaves `1` as `1`. It therefore interprets `-1` as left and `1` as right.

ii. ```python
if EXCLUDE_NOCHOICE:
    query_parts.append('(choice == 0)')
```

```python
choice = final_trials['choice'].values.copy()
choice[choice == -1] = 0
choice = choice.astype(np.float32)
```

iii. The notes explicitly justify this interpretation as 'Choice: -1 (left) -> 0, 1 (right) -> 1'.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trial table's `probabilityLeft` column.

ii. ```python
prob_left = final_trials['probabilityLeft'].values.copy()
```

iii. The notes map block prior directly from `probabilityLeft`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The script maps `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2` with `np.isclose` comparisons after trial filtering.

ii. ```python
prior = np.zeros(len(prob_left), dtype=np.float32)
prior[np.isclose(prob_left, 0.2)] = 0
prior[np.isclose(prob_left, 0.5)] = 1
prior[np.isclose(prob_left, 0.8)] = 2
```

iii. The notes justify keeping unbiased trials specifically so the `0.5` prior class can be represented.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. ```python
pos_file = find_file(alf_dir, '_ibl_wheel.position.npy')
ts_file = find_file(alf_dir, '_ibl_wheel.timestamps.npy')
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The notes state that wheel speed should come from the wheel position stream in the IBL cache.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The script estimates the sampling frequency from the median timestamp difference, applies `brainbox.behavior.wheel.velocity_filtered(position, fs)` when possible, falls back to a finite-difference derivative otherwise, takes the absolute value as speed, and then linearly interpolates the resulting speed trace onto each trial's 100-bin grid.

ii. ```python
dt_median = np.median(np.diff(timestamps))
if dt_median <= 0:
    dt_median = 0.001
fs = 1.0 / dt_median
try:
    velocity, _ = velocity_filtered(position, fs)
except Exception:
    dt = np.diff(timestamps)
    dt[dt == 0] = dt_median
    velocity = np.zeros_like(position)
    velocity[1:] = np.diff(position) / dt
    velocity[0] = velocity[1]
speed = np.abs(velocity)
```

```python
wheel_binned, wheel_valid = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, interval_begs, interval_ends, BINSIZE, N_BINS
)
```

iii. Trajectory Steps 189, 193, 196, and 204 show the AI explicitly revising an earlier finite-difference implementation after realizing the reference used `SessionLoader.load_wheel()` and `velocity_filtered`. Its justification was to better match the reference processing.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. It is discretized globally across all retained sessions, not per session. After all sessions are processed, the script pools every wheel-speed bin value, computes the 33.33rd and 66.67th percentiles, and applies `np.digitize` to each session's wheel trace with those two thresholds.

ii. ```python
all_wheel_vals = np.concatenate([w.flatten() for w in all_wheel_raw])
wheel_percentiles = np.percentile(all_wheel_vals[~np.isnan(all_wheel_vals)], [33.33, 66.67])
...
wheel_disc = np.digitize(wheel_raw, wheel_percentiles).astype(np.int64)
```

iii. The notes state this was a deliberate Step 5 choice: 'Wheel speed discretization: 3 equal-frequency bins across all data.'

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. For each trial, the wheel trace is interpolated to `n_bins` points defined by `np.linspace(t_beg + binsize, t_end, n_bins)`, where `t_beg = stimOn_times - 0.5` and `t_end = stimOn_times + 1.5`. The result is then kept only if interpolation succeeded for that trial.

ii. ```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
...
interp_func = interp1d(local_times, local_vals, kind='linear',
                       fill_value='extrapolate')
result[trial_idx] = interp_func(x_interp).astype(np.float32)
```

iii. The notes justify this as 'Behavior interpolation: linear interpolation to match neural time bins', although the implementation uses its own interpolation grid rather than the exact loader logic.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` with `_ibl_leftCamera.times.npy` if available, otherwise from the analogous right-camera files.

ii. ```python
me_file = find_file(alf_dir, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_dir, '_ibl_leftCamera.times.npy')
if me_file is None or times_file is None:
    me_file = find_file(alf_dir, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_file(alf_dir, '_ibl_rightCamera.times.npy')
```

iii. The notes say this was chosen to mirror the reference's 'left camera first, fallback to right' behavior.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The script loads the released motion-energy trace as-is, trims the values and timestamps to the same minimum length, removes any NaN samples, and later interpolates the cleaned trace onto each trial grid. It does not apply extra filtering or normalization before discretization.

ii. ```python
me_values = np.load(me_file).flatten()
me_times = np.load(times_file).flatten()
min_len = min(len(me_values), len(me_times))
me_values = me_values[:min_len]
me_times = me_times[:min_len]
valid = ~(np.isnan(me_values) | np.isnan(me_times))
me_values = me_values[valid]
me_times = me_times[valid]
```

iii. The notes justify this mostly as robust handling of missing or malformed samples while preserving the raw whisker-energy signal.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, it is discretized globally across all retained sessions. The script pools all whisker-motion values after interpolation, computes global 33.33rd and 66.67th percentiles, then digitizes each session's whisker trace with those thresholds.

ii. ```python
all_me_vals = np.concatenate([m.flatten() for m in all_me_raw])
me_percentiles = np.percentile(all_me_vals[~np.isnan(all_me_vals)], [33.33, 66.67])
...
me_disc = np.digitize(me_raw, me_percentiles).astype(np.int64)
```

iii. The notes explicitly say 'Whisker ME discretization: 3 equal-frequency bins across all data.'

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned the same way as wheel speed: trial-by-trial linear interpolation onto `np.linspace(t_beg + binsize, t_end, n_bins)` using the trial's stimulus-locked window.

ii. ```python
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, interval_begs, interval_ends, BINSIZE, N_BINS
)
```

```python
x_interp = np.linspace(t_beg + binsize, t_end, n_bins)
interp_func = interp1d(local_times, local_vals, kind='linear',
                       fill_value='extrapolate')
result[trial_idx] = interp_func(x_interp).astype(np.float32)
```

iii. The notes describe this as matching the neural time bins by interpolation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code is defensive but coarse. Missing trial-table files or missing spike, wheel, or whisker files cause the whole session to be skipped. Whisker traces with mismatched value/time lengths are truncated to the shorter length, NaN whisker samples are dropped, bad wheel timestamps fall back to a default `dt`, wheel-velocity filtering falls back to finite differences if needed, and sessions with fewer than two valid trials after interpolation are dropped.

ii. ```python
if trials_df is None:
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
min_len = min(len(me_values), len(me_times))
me_values = me_values[:min_len]
me_times = me_times[:min_len]
valid = ~(np.isnan(me_values) | np.isnan(me_times))
```

```python
if n_final < 2:
    print(f'  Too few valid trials after behavior filtering')
    return None
```

iii. The notes justify these choices as practical handling of edge cases in a large cache. Step 9 and Step 10 of `CONVERSION_NOTES.md` specifically say sessions were skipped for missing whisker motion energy and that NaN or length mismatches were handled explicitly.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive parts are reading large spike arrays from disk for each probe, per-trial spike binning over those arrays, and finally writing the very large pickle after all-session aggregation. Behavior interpolation is comparatively lighter but still repeated trial-by-trial.

ii. ```python
spike_times = np.load(os.path.join(version_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(version_dir, 'spikes.clusters.npy')).flatten()
```

```python
binned_spikes = bin_spikes_fast(
    spike_times, spike_clusters, n_clusters,
    interval_begs, interval_ends, BINSIZE, N_BINS
)
```

```python
with open(args.output, 'wb') as f:
    pickle.dump(data, f)
```

iii. The notes estimate about 37 minutes for full conversion and log per-session spike-binning timing, indicating the agent was watching I/O and binning costs.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are obvious vectorization targets: the per-trial loop in `bin_spikes_fast`, the per-trial interpolation loop in `interpolate_behavior_to_bins`, the scan over all trials in `compute_trial_number_in_block`, the per-trial construction of `neural`/`input` lists, and the second per-session/per-trial pass that rebuilds outputs after global discretization.

ii. ```python
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
    if prob_left[i] != current_prob:
        ...
```

```python
for t in range(n_final_trials):
    neural_list.append(final_spikes[t].astype(np.float32))
    input_list.append(inp)
```

```python
for sess_i in range(len(all_neural)):
    ...
    for t in range(n_trials):
        out = np.stack([...]).astype(np.int64)
        output_list.append(out)
```

iii. The notes say efficiency mattered and mention vectorization as a goal, but the final implementation kept several explicit Python loops for clarity and convenience.

## 10-c. What processing does the code repeat multiple times?

i. It repeats trial-wise interpolation logic separately for wheel and whisker traces, and it performs output construction in two stages: first it stores all continuous wheel and whisker traces session-by-session, then it loops back over every session and every trial to discretize them and build the final `output` arrays.

ii. ```python
wheel_binned, wheel_valid = interpolate_behavior_to_bins(...)
me_binned, me_valid = interpolate_behavior_to_bins(...)
```

```python
all_wheel_raw.append(result['wheel_speed_raw'])
all_me_raw.append(result['me_raw'])
...
for sess_i in range(len(all_neural)):
    ...
    wheel_disc = np.digitize(wheel_raw, wheel_percentiles).astype(np.int64)
    me_disc = np.digitize(me_raw, me_percentiles).astype(np.int64)
    for t in range(n_trials):
        out = np.stack([...]).astype(np.int64)
```

iii. The notes justify the second pass by the decision to use global discretization thresholds, which requires keeping all raw wheel and whisker traces until every session has been processed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It allocates `output_list_wheel` and `output_list_me` but never uses them, stores all raw wheel and whisker traces for every kept trial solely to compute global thresholds later, and carries some metadata such as `lab` that is never used in the final dataset. Those intermediate continuous traces are discarded once the discretized outputs are built.

ii. ```python
output_list_wheel = []
output_list_me = []
```

```python
return {
    ...
    'wheel_speed_raw': final_wheel,
    'me_raw': final_me,
    ...
}
```

```python
lab = group['lab'].iloc[0]
...
sessions_info.append({
    'eid': eid,
    'subject': subject,
    'date': date,
    'lab': lab,
    ...
})
```

iii. The notes frame the extra raw-trace retention as necessary for the chosen global discretization scheme, but those raw arrays do not survive into the final `output` field.
