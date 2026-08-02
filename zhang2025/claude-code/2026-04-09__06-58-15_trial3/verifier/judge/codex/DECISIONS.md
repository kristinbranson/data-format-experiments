# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the BWM session manifest from `bwm_release.csv`, groups rows by `eid`, then iterates over each session on disk under `/app/data/one_cache/{lab}/Subjects/{subject}/{date}/{session_number}`. Within each session it loads trials, all listed probes, wheel data, and whisker motion energy directly from ALF files rather than through ONE.

ii. ```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
session_groups = bwm_df.groupby('eid')
for eid, group in session_groups:
    row = group.iloc[0]
    probe_names = list(group['probe_name'])
    session_list.append((eid, row['lab'], row['subject'], row['date'],
                         row['session_number'], probe_names))
...
session_path = os.path.join(DATA_ROOT, lab, 'Subjects', subject, date, session_num_str)
trials_df = load_trials(session_path)
for pname in probe_names:
    spikes, clusters = load_spike_data(session_path, pname)
wheel_times, wheel_speed = load_wheel_speed(session_path)
me_times, me_values = load_whisker_motion_energy(session_path)
```

iii. In `CONVERSION_NOTES.md`, the agent says it followed `0_data_caching.py` semantically but used direct disk loading because the cache is read-only and local data are already present.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` field in each grouped session entry. The final `subjects` list is built incrementally in first-seen order over processed sessions, and each session gets a `subject_idx` pointing into that list.

ii. ```python
subject_set = []
...
subject = result['subject']
if subject not in subject_set:
    subject_set.append(subject)
subject_idx = subject_set.index(subject)
...
'subjects': subject_set,
'subject_idx': np.array(all_subject_idx, dtype=np.int64),
```

iii. The notes describe the dataset as organized by `{lab}/Subjects/{subject}/...` and explicitly track subject counts from `bwm_release.csv`.

## 1-c. How are the data split into sessions?

i. Sessions are split by unique `eid` values in `bwm_release.csv`. Multiple probes for the same `eid` are merged into one session record, matching the reference pipeline’s session-level processing.

ii. ```python
session_groups = bwm_df.groupby('eid')
for eid, group in session_groups:
    row = group.iloc[0]
    probe_names = list(group['probe_name'])
    session_list.append((eid, row['lab'], row['subject'], row['date'],
                         row['session_number'], probe_names))
```

iii. The notes explicitly cite `prepare_data()` and `merge_probes()` and say the goal is one merged dataset per session, not one per probe.

## 1-d. How are the data split into trials?

i. Trials come from rows of `_ibl_trials.table.pqt`. Each row defines one trial, and per-trial intervals are built from `stimOn_times + (-0.5, 1.5)` for spike and behavior extraction.

ii. ```python
def load_trials(session_path):
    trials_file = find_file(alf_path, '_ibl_trials.table.pqt')
    return pd.read_parquet(trials_file)
...
stim_times = trials_df[ALIGN_TIME].values
interval_begs = stim_times + TIME_WINDOW[0]
interval_ends = stim_times + TIME_WINDOW[1]
```

iii. The notes say this matches `bin_spiking_data()` in Zhang’s code with `align_time='stimOn_times'` and `time_window=(-0.5, 1.5)`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered with a boolean mask that removes NaNs in key trial fields, RTs outside `[0.08, 2.0]`, trials longer than 10 s, and no-choice trials. The agent then further requires wheel and whisker interpolation to succeed for the aligned interval.

ii. ```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']
...
mask &= ~trials_df[event].isna()
rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
mask &= (rt >= MIN_RT)
mask &= (rt <= MAX_RT)
trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()
mask &= (trials_df['choice'] != 0)
...
combined_mask = mask & wheel_mask & me_mask
```

iii. The notes cite `load_trials_and_mask()` defaults from the reference code and say behavior masks are also applied so neural and behavioral streams remain aligned.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from per-probe `spikes.times.npy` and `spikes.clusters.npy`, with cluster metadata loaded from `clusters.channels.npy`, `clusters.depths.npy`, `clusters.metrics.pqt`, and channel/brain-location arrays for region labels.

ii. ```python
times_file = os.path.join(spike_dir, 'spikes.times.npy')
clusters_file = os.path.join(spike_dir, 'spikes.clusters.npy')
spikes = {
    'times': np.load(times_file).flatten(),
    'clusters': np.load(clusters_file).flatten(),
}
channels_file = os.path.join(spike_dir, 'clusters.channels.npy')
depths_file = os.path.join(spike_dir, 'clusters.depths.npy')
```

iii. The notes identify `load_spiking_data()` and `merge_probes()` as the reference functions and state that all clusters are used.

## 2-b. How is the `neural` data processed?

i. Probe spike trains are merged by offsetting cluster ids and time-sorting. Then, for each trial window, spikes are counted in 20 ms bins with `np.bincount` over linearized `(cluster, bin)` indices, yielding `(n_trials, n_clusters, 100)` arrays that are later stored per trial as `(n_neurons, n_timepoints)`.

ii. ```python
merged_spikes, cluster_brain_ids = merge_probes(spikes_list, clusters_list)
...
binned_spikes = bin_spikes_fast(
    merged_spikes['times'], merged_spikes['clusters'],
    interval_begs, interval_ends, n_clusters
)
...
lin_idx = c * N_BINS + b
counts = np.bincount(lin_idx, minlength=minlength)
binned[trial_idx] = counts[:minlength].reshape(n_clusters_total, N_BINS)
```

iii. The notes say this is an optimized reimplementation of `get_spike_data_per_interval()` / `bin_spiking_data()` from the Zhang code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered by neuron quality. The agent keeps all clusters from spike sorting and applies no QC threshold on cluster labels.

ii. ```python
# No neuron quality filtering: all clusters in merged_spikes are retained
n_clusters = int(merged_spikes['clusters'].max()) + 1 if len(merged_spikes['clusters']) > 0 else 0
...
'neuron_filtering': 'none (all clusters used, matching Zhang et al. 2025)',
```

iii. The notes explicitly justify this with the reference code’s `qc=None` default in `load_spiking_data()` despite the data paper’s separate “well-isolated neuron” statistics.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to stimulus onset. The interval is `[stimOn_times - 0.5, stimOn_times + 1.5]` seconds, and spikes in that interval are binned.

ii. ```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)
...
stim_times = trials_df[ALIGN_TIME].values
interval_begs = stim_times + TIME_WINDOW[0]
interval_ends = stim_times + TIME_WINDOW[1]
```

iii. The notes say this resolves the paper/code discrepancy by following the task instruction and the unified alignment used in `0_data_caching.py`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins, giving 100 bins over a 2 s window. No later temporal rebinning is applied.

ii. ```python
BINSIZE = 0.02
INTERVAL_LEN = TIME_WINDOW[1] - TIME_WINDOW[0]
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The notes explicitly say the agent followed `binsize = 0.02` from `0_data_caching.py`, not the analysis-specific binning descriptions in the paper text.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read from a single raw data array. It is constructed from the alignment specification itself: `stimOn_times` defines the zero point, while `TIME_WINDOW` and `BINSIZE` define the relative per-bin time vector.

ii. ```python
time_since_onset = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The notes describe this as “computed from bin edges” and intended to match the interpolation grid used for behaviors.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The agent creates one fixed 100-point vector from `-0.48` to `1.5` s, corresponding to the right edge of each 20 ms bin in the aligned trial window, and copies it into every trial’s first input row.

ii. ```python
time_since_onset = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
...
input_trial[0, :] = time_since_onset
```

iii. The notes say this follows the reference behavior interpolation convention `np.linspace(interval_beg + binsize, interval_end, n_bins)`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the same 100-bin grid as neural activity, over the same `stimOn_times`-aligned window. Every time value corresponds to the same bin positions used for spikes and interpolated behaviors.

ii. ```python
input_trial = np.zeros((2, N_BINS), dtype=np.float32)
input_trial[0, :] = time_since_onset
...
neural_trial = neural_data[trial_idx].astype(np.uint8)
```

iii. The notes explicitly state that time since onset is on the same grid as the neural bins.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column in the trials table, using changes in that value to define block boundaries.

ii. ```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
trial_num_in_block = all_trial_nums[good_indices].astype(np.float32)
```

iii. The notes say this is a custom variable required by the task and inferred from the block structure encoded in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The agent scans the full unfiltered trial sequence, resets a block-start index whenever `probabilityLeft` changes, and assigns each trial a zero-based offset from the current block start. Afterward it indexes that sequence by the final kept trials.

ii. ```python
def compute_trial_number_in_block(prob_left):
    trial_numbers = np.zeros(len(prob_left), dtype=np.float32)
    current_block_start = 0
    current_val = prob_left[0]
    for i in range(len(prob_left)):
        if prob_left[i] != current_val:
            current_block_start = i
            current_val = prob_left[i]
        trial_numbers[i] = i - current_block_start
```

iii. The notes justify computing this before filtering so block transitions are not distorted by dropped trials.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the `choice` column of the trials table.

ii. ```python
choice_raw = trials_df['choice'].values[good_indices]
```

iii. The notes identify `trials.choice` as the source and map it to the task’s binary target.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent remaps IBL choice coding from `-1/1` to `0/1`, then broadcasts the per-trial class label across all 100 time bins in each output trial matrix.

ii. ```python
choice = ((choice_raw + 1) / 2).astype(np.int32)
...
output_trial[0, :] = choice[trial_idx]
```

iii. The notes explicitly state “-1→0 (left), 1→1 (right)” to satisfy the decoder task.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column of the trials table.

ii. ```python
prob_left = trials_df['probabilityLeft'].values[good_indices]
```

iii. The notes say this corresponds to Zhang’s `block` variable and is remapped to the task’s named output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The agent converts `probabilityLeft` values `0.2`, `0.5`, `0.8` into integer classes `0`, `1`, `2`, then broadcasts the per-trial class across the full time axis.

ii. ```python
prior = np.zeros(len(prob_left), dtype=np.int32)
prior[prob_left == 0.2] = 0
prior[prob_left == 0.5] = 1
prior[prob_left == 0.8] = 2
...
output_trial[1, :] = prior[trial_idx]
```

iii. The notes list this exact mapping in the “Variable Mapping” section.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from raw wheel position and timestamp arrays: `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. ```python
pos_file = find_file(alf_path, '_ibl_wheel.position.npy')
ts_file = find_file(alf_path, '_ibl_wheel.timestamps.npy')
re_pos = np.load(pos_file).flatten()
re_ts = np.load(ts_file).flatten()
```

iii. The notes say this reproduces `load_target_behavior(one, eid, 'wheel-speed')` without using ONE.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The raw wheel position is interpolated to a uniform 1000 Hz grid, low-pass filtered with an 8th-order 20 Hz Butterworth filter, differentiated to get velocity, converted to absolute speed, then linearly interpolated onto each trial’s 100 aligned bins.

ii. ```python
position = scipy_interp1d(re_ts, re_pos, kind='linear')(t)
sos = scipy.signal.butter(N=order, Wn=corner_frequency / fs * 2,
                          btype='lowpass', output='sos')
vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, position)), 0, 0) * fs
speed = np.abs(vel).astype(np.float32)
...
binned_wheel, wheel_mask = interpolate_behavior(
    wheel_times, wheel_speed, interval_begs, interval_ends)
```

iii. The notes say this was chosen to match `SessionLoader.load_wheel()` / `velocity_filtered`.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. After trial alignment, the agent flattens all retained wheel-speed values within a session and discretizes them into three equal-frequency bins using percentile thresholds. It then reshapes those labels back to `(n_trials, 100)`.

ii. ```python
wheel_flat = wheel_data.flatten()
wheel_discrete = discretize_to_bins(wheel_flat, n_bins=3)
wheel_discrete_2d = wheel_discrete.reshape(wheel_data.shape).astype(np.int32)
```

iii. The notes justify this as a practical way to satisfy the task’s “3 bins” requirement, since the reference code keeps wheel speed continuous.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same `stimOn_times` window as neural data. For each trial, the continuous signal is interpolated to `np.linspace(interval_beg + binsize, interval_end, 100)`.

ii. ```python
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
y_interp = interp1d(trial_times, trial_vals, kind='linear',
                    fill_value='extrapolate')(x_interp)
```

iii. The notes explicitly say all variables use the unified stimulus-onset alignment from `0_data_caching.py`.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` with `_ibl_leftCamera.times.npy`, with fallback to `rightCamera.ROIMotionEnergy.npy` and `_ibl_rightCamera.times.npy` if left-camera data are unavailable.

ii. ```python
me_file = find_file(alf_path, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_path, '_ibl_leftCamera.times.npy')
...
me_file = find_file(alf_path, 'rightCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_path, '_ibl_rightCamera.times.npy')
```

iii. The notes say this mirrors the reference behavior loader’s left-then-right fallback.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The agent loads the ROI motion-energy trace and timestamps from camera files, then linearly interpolates that continuous signal onto the same 100 per-trial aligned bins used for neural data.

ii. ```python
me_times, me_values = load_whisker_motion_energy(session_path)
...
binned_me, me_mask = interpolate_behavior(
    me_times, me_values, interval_begs, interval_ends)
```

iii. The notes refer to this as matching `load_target_behavior(... whisker-motion-energy)` plus `get_behavior_per_interval()`.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, whisker motion energy is flattened across all retained bins in a session and discretized into three equal-frequency bins by percentile thresholds, then reshaped back into per-trial time series.

ii. ```python
me_flat = me_data.flatten()
me_discrete = discretize_to_bins(me_flat, n_bins=3)
me_discrete_2d = me_discrete.reshape(me_data.shape).astype(np.int32)
```

iii. The notes use the same justification as for wheel speed: the reference pipeline keeps the variable continuous, so the converter chose session-wise quantile bins to meet the task format.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned exactly like wheel speed and spikes: same `stimOn_times` anchor, same `[-0.5, 1.5]` window, same 100 interpolation targets.

ii. ```python
combined_mask = mask & wheel_mask & me_mask
...
output_trial[3, :] = me_discrete_2d[trial_idx]
```

iii. The notes repeatedly state that the whole converted dataset uses a single stimulus-onset-aligned time base.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing files or malformed arrays cause the corresponding loader to return `None`, which can skip a behavior stream or whole session. Missing/NaN trial values are excluded by the trial mask. Trials whose behavior intervals cannot be cleanly interpolated are dropped via `wheel_mask`/`me_mask`. Unexpected exceptions skip the session.

ii. ```python
if trials_file is None:
    return None
...
if pos_file is None or ts_file is None:
    return None, None
...
if len(trial_vals) == 0:
    continue
if np.abs(t_beg - trial_times[0]) > BINSIZE:
    continue
...
except Exception as e:
    print(f"  ERROR processing {eid}: {type(e).__name__}: {e}")
    n_skipped += 1
    continue
```

iii. The notes frame this as matching the reference code’s practical behavior: skip bad sessions/trials rather than repairing data.

## 12-a. What are the most time-consuming steps of the code?

i. The agent identifies spike binning as the dominant cost, followed by loading/interpolating behavior traces and final reassembly/pickling of all sessions. Wheel-speed preprocessing is also nontrivial because it resamples to 1000 Hz before alignment.

ii. ```python
t_spike = time.time()
binned_spikes = bin_spikes_fast(...)
t_spike_done = time.time()
...
t_beh_done = time.time()
...
print(f"  {eid}: {n_trials} trials, {n_clusters} neurons | "
      f"spike_bin={t_spike_done-t_spike:.1f}s, beh={t_beh_done-t_spike_done:.1f}s, "
      f"total={t_format_done-t0:.1f}s")
```

iii. The notes explicitly say spike binning was optimized because it was the major bottleneck.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still has several Python-level loops: trial-by-trial spike binning, spike-by-spike accumulation inside `bin_spikes_vectorized`, session-by-session and neuron-by-neuron metadata bookkeeping, trial-by-trial formatting into Python lists, and the block-index loop.

ii. ```python
for i, trial_idx in enumerate(valid_idx):
    ...
for spike_i in range(len(trial_times)):
    c = trial_clusters[spike_i]
    b = bin_idx[spike_i]
...
for neuron_idx in range(result['n_neurons']):
    region = result['cluster_beryl'][neuron_idx]
...
for trial_idx in range(n_trials):
    neural_list.append(neural_trial)
```

iii. The notes mention some vectorization already added, but the remaining loops are still straightforward improvement targets.

## 12-c. What processing does the code repeat multiple times?

i. It repeats several computations: RT is recomputed twice in the trial mask; wheel and whisker discretization use the same flatten/discretize/reshape pattern; temp session files are loaded once for metadata and again for final assembly; per-trial broadcasting work is repeated separately for each output/input tensor.

ii. ```python
rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
mask &= (rt >= MIN_RT)
...
rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
mask &= (rt <= MAX_RT)
...
with open(meta['tmp_file'], 'rb') as f:
    sess = pickle.load(f)
...
with open(session_meta[idx]['tmp_file'], 'rb') as f:
    sess = pickle.load(f)
```

iii. The notes emphasize runtime optimizations, and these repeated steps are visible in the final script despite those optimizations.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads cluster depths and metrics even though downstream decoder training only uses spike counts and brain-region indices. It also builds `merged_channels` and `merged_depths` in `merge_probes()` without returning them, and optional plotting work is not used by the saved dataset.

ii. ```python
if os.path.exists(depths_file):
    clusters['depths'] = np.load(depths_file).flatten()
if os.path.exists(metrics_file):
    clusters['metrics'] = pd.read_parquet(metrics_file)
...
merged_channels = []
merged_depths = []
...
if show_processing:
    plot_processing(...)
```

iii. The notes describe these as part of matching reference loading or debugging, not as requirements for the final decoder input.
