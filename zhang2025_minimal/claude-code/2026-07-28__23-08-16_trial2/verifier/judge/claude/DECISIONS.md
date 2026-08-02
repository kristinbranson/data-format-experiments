# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the BWM release CSV (`bwm_release.csv`) to get the list of all probe insertions (PIDs) and session IDs (eids). It iterates over unique eids. For each session, it loads spike-sorting data per probe via `SpikeSortingLoader`, loads trials via `SessionLoader.load_trials()`, loads wheel data via `SessionLoader.load_wheel()`, and loads whisker motion energy via `SessionLoader.load_motion_energy()`. All loading is done through the ONE API connected to the IBL Alyx database.

ii.
```python
bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)
eids = bwm_df['eid'].unique()

for eid_idx, eid in enumerate(eids):
    rows = bwm_df[bwm_df['eid'] == eid]
    # ...load spike sorting per probe...
    for pid, pname in zip(pids, probe_names):
        spks, clust = load_spiking_data(one, pid, eid=eid, pname=pname)
    # ...load trials...
    sl = SessionLoader(one=one, eid=eid)
    sl.load_trials()
    # ...load wheel...
    sl.load_wheel()
    # ...load whisker ME...
    sl.load_motion_energy(views=['left'])
```

iii. The AI states it follows the reference code's `prepare_data` pipeline: "Load spike sorted data per probe, merge probes, load trials and apply mask, load behavioral data." The BWM release CSV is the same file used by the reference code.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `subject` column of the BWM release DataFrame. A mapping from subject name to integer index is built incrementally as sessions are processed. The `subjects` list and `subject_idx` array are stored in the output.

ii.
```python
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects_list)
    subjects_list.append(subject)
subj_idx = subject_to_idx[subject]
```

iii. The AI notes that there are 139 unique subjects, matching the paper's statement "We trained 139 mice."

## 1-c. How are the data split into sessions?

i. Each unique `eid` in the BWM release DataFrame constitutes one session. Sessions are processed independently, and data from each session is stored as a separate entry in the `neural`, `input`, and `output` lists.

ii.
```python
eids = bwm_df['eid'].unique()
for eid_idx, eid in enumerate(eids):
    # ... process entire session ...
    neural_list.append(session_neural)
    input_list.append(session_input)
    output_list.append(session_output)
```

iii. The AI reports processing 444 sessions (out of 459 in the BWM release), with 15 skipped due to missing whisker motion energy data. The reference paper uses 433 sessions.

## 1-d. How are the data split into trials?

i. Within each session, trials are loaded from the IBL trials table. A trial mask is applied to exclude invalid trials. Valid trials are then individually binned into 2-second windows aligned to stimulus onset. Each trial becomes a separate entry in the session's list of trials.

ii.
```python
masked_trials = trials[mask].reset_index(drop=True)
align_times = masked_trials['stimOn_times'].values
binned_spikes = bin_spikes_in_window(
    spikes['times'], spikes['clusters'], cluster_ids,
    align_times, WINDOW, BINSIZE
)
```

iii. The AI states: "Trials are split based on the IBL trial structure. Each trial is a 2-second window aligned to stimulus onset."

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if: (1) reaction time (firstMovement_times - stimOn_times) is outside [0.08, 2.0]s; (2) any of six key events are NaN (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType); (3) the animal made no choice (choice == 0). Additionally, trials where wheel speed or whisker motion energy data is unavailable or has poor coverage are excluded via a `combined_mask`.

ii.
```python
def load_trials_and_mask(one, eid, min_rt=0.08, max_rt=2.0):
    nan_exclude = [
        'stimOn_times', 'choice', 'feedback_times',
        'probabilityLeft', 'firstMovement_times', 'feedbackType'
    ]
    mask = pd.Series(True, index=trials.index)
    rt = trials['firstMovement_times'] - trials['stimOn_times']
    mask &= rt >= min_rt
    mask &= rt <= max_rt
    for event in nan_exclude:
        mask &= ~trials[event].isnull()
    mask &= trials['choice'] != 0
```

iii. The AI states this matches the reference code's default `load_trials_and_mask` parameters. Note: the reference code in `prepare_data` also passes `max_trial_len=10.0` which the AI does not include.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times (`spikes['times']`) and spike cluster assignments (`spikes['clusters']`) loaded via `SpikeSortingLoader`. All spike-sorted clusters are used (no quality control filter, `qc=None`).

ii.
```python
ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = ssl.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
return spikes, clusters_labeled
```

iii. The AI states: "Use ALL spike-sorted clusters (no quality control filtering) matching the reference code's default behavior in prepare_data." This matches the reference code where `load_spiking_data` is called without the `qc` parameter.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms non-overlapping bins within a [-0.5, 1.5]s window relative to stimulus onset. For multi-probe sessions, spikes from all probes are merged before binning. The result is spike counts per neuron per time bin, shape (n_neurons, n_bins=100).

ii.
```python
WINDOW = (-0.5, 1.5)
BINSIZE = 0.02
# ...
binned_spikes = bin_spikes_in_window(
    spikes['times'], spikes['clusters'], cluster_ids,
    align_times, WINDOW, BINSIZE
)
# bin_spikes_in_window uses searchsorted + np.add.at for accumulation
bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
np.clip(bin_indices, 0, n_bins - 1, out=bin_indices)
np.add.at(binned[trial_idx], (cluster_indices[valid], bin_indices[valid]), 1)
```

iii. The AI states: "20ms non-overlapping bins -> 100 time steps per 2s trial" matching the reference code's `binsize=0.02` and `time_window=(-.5, 1.5)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No quality control filtering is applied to neurons. All spike-sorted clusters are retained (including multi-unit activity). The AI uses `qc=None` (the default), meaning all clusters pass.

ii.
```python
def load_spiking_data(one, pid, eid='', pname=''):
    """Uses all clusters (no QC filter)"""
    ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
    return spikes, clusters_labeled
```

iii. The AI states: "The reference code's prepare_data function calls load_spiking_data without the qc parameter, defaulting to qc=None which returns all clusters." This matches the reference code. However, the AI includes ALL clusters (even those with zero spikes throughout the recording) in the neural matrix, whereas the reference code's `bin_spiking_data` only includes clusters that have at least one spike.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, the window is [stimOn_times - 0.5, stimOn_times + 1.5] seconds. Spikes within this window are binned into 100 time bins of 20ms each.

ii.
```python
align_times = masked_trials['stimOn_times'].values
# In bin_spikes_in_window:
t_start = align_times[trial_idx] + window[0]  # stimOn - 0.5
t_end = align_times[trial_idx] + window[1]    # stimOn + 1.5
```

iii. The AI states: "Methods paper states: 'For choice, we align trials to the stimulus onset'... The reference code uses params = {'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}." This matches the task instructions: "Temporally align based on stimulus onset."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20ms (0.02s), producing 100 time bins per 2-second trial. No additional temporal rebinning is applied.

ii.
```python
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.round((WINDOW[1] - WINDOW[0]) / BINSIZE))  # = 100
```

iii. The AI notes the methods paper mentions 50ms bins for choice/prior decoding, but follows the reference code which uses 20ms universally: "We follow the reference code since we decode all variables simultaneously."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Time since stimulus onset is not derived from raw data variables per trial. It is constructed as a fixed time series representing the bin centers of the trial window, computed from the window parameters and bin size.

ii.
```python
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)
# Results in values from -0.48 to 1.5 in 100 steps
```

iii. The AI states: "Time since stimulus onset: Continuous time variable, same for all trials (linspace from -0.48 to 1.5, matching bin centers)."

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The input is a linearly spaced vector from (window_start + binsize) to window_end, with N_BINS points. This corresponds to bin centers of the trial window. The values range from -0.48s to 1.5s.

ii.
```python
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The AI states this matches the reference code's bin center computation: `np.linspace(interval_begs[interval_idx] + binsize, interval_ends[interval_idx], n_bins)`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time since stimulus onset vector uses the same bin centers as the neural data and behavioral interpolation. Each element of the time vector corresponds to one column (time bin) in the neural activity matrix.

ii.
```python
# Same formula used for bin centers in both neural binning and behavior interpolation:
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
# And for input:
time_since_stim = np.linspace(WINDOW[0] + BINSIZE, WINDOW[1], N_BINS)
```

iii. Alignment is implicit since the same bin center formula is used everywhere.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from the `probabilityLeft` column of the trials table. Block boundaries are detected where `probabilityLeft` changes from one trial to the next.

ii.
```python
def compute_trial_number_in_block(trials_df, mask):
    pLeft = trials_df['probabilityLeft'].values
    trial_num = np.zeros(len(pLeft), dtype=np.float32)
    block_counter = 1
    for i in range(len(pLeft)):
        if i == 0 or pLeft[i] != pLeft[i-1]:
            block_counter = 1
        trial_num[i] = block_counter
        block_counter += 1
    return trial_num[mask]
```

iii. The AI states: "A block change occurs when probabilityLeft changes from one trial to the next. Trial number is 1-indexed within each block."

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The algorithm iterates over all trials in order, resetting a counter to 1 whenever `probabilityLeft` changes. The counter increments by 1 for each subsequent trial in the same block. The result is then filtered to only include masked (valid) trials. The per-trial value is replicated across all time bins.

ii.
```python
# Per-trial value replicated across time bins:
trial_num = np.full(N_BINS, sm['trial_num_in_block'][trial], dtype=np.float32)
inp = np.stack([time_input, trial_num], axis=0)  # (2, T)
```

iii. The AI notes this is 1-indexed and computed over ALL trials (including invalid ones) before filtering, so the count accurately reflects the trial's position within its block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the IBL trials table.

ii.
```python
choice_vals = masked_trials.iloc[masked_idx]['choice'].values.copy()
choice = np.where(choice_vals == 1, 1, 0).astype(np.int64)
```

iii. The AI states: "In IBL data, choice=-1 is left, choice=1 is right." However, the actual IBL convention is choice=1 for LEFT and choice=-1 for RIGHT (confirmed by `brainbox/behavior/training.py:590` where `rightward = trials.choice == -1`).

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps IBL choice values to binary: choice==1 maps to 1, everything else (choice==-1) maps to 0. The per-trial choice is then replicated across all 100 time bins to create a time-varying output.

ii.
```python
choice = np.where(choice_vals == 1, 1, 0).astype(np.int64)
# Then in the output construction:
choice_arr = np.full(N_BINS, sm['choice'][trial], dtype=np.int64)
```

iii. The AI states: "Choice: left=-1 -> 0, right=1 -> 1 (as per decoder spec)". However, the IBL convention is choice=1 for LEFT and choice=-1 for RIGHT. The AI has the mapping inverted: LEFT (choice=1) is mapped to 1 (should be 0) and RIGHT (choice=-1) is mapped to 0 (should be 1).

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior is derived from the `probabilityLeft` column of the IBL trials table.

ii.
```python
pLeft = masked_trials.iloc[masked_idx]['probabilityLeft'].values
prior = np.zeros(len(pLeft), dtype=np.int64)
prior[pLeft == 0.2] = 0
prior[pLeft == 0.5] = 1
prior[pLeft == 0.8] = 2
```

iii. The AI states: "Decoder task specifies '0.2 -> 0, 0.5 -> 1, 0.8 -> 2'" which matches the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The continuous `probabilityLeft` values (0.2, 0.5, 0.8) are mapped to categorical integers (0, 1, 2). The per-trial value is replicated across all 100 time bins.

ii.
```python
prior[pLeft == 0.2] = 0
prior[pLeft == 0.5] = 1
prior[pLeft == 0.8] = 2
# Then:
prior_arr = np.full(N_BINS, sm['prior'][trial], dtype=np.int64)
```

iii. The AI states: "3-class categorical: 0.2->0, 0.5->1, 0.8->2" matching the decoder task specification.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from the `velocity` column of `SessionLoader.wheel`, which contains wheel velocity computed with Gaussian smoothing. The absolute value is taken to convert velocity to speed.

ii.
```python
sl.load_wheel()
wheel_times = sl.wheel['times'].values
wheel_speed = np.abs(sl.wheel['velocity'].values)
```

iii. The AI states: "Reference code's load_target_behavior for 'wheel-speed' computes np.abs(sess_loader.wheel['velocity'].to_numpy())" which matches the reference code exactly.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. Absolute velocity is computed session-wide. Then, for each trial, the speed is linearly interpolated to bin centers within the trial window using `scipy.interpolate.interp1d`. The bin centers are computed as `np.linspace(t_start + binsize, t_end, n_bins)`.

ii.
```python
def interpolate_behavior_to_bins(beh_times, beh_values, align_times, window, binsize):
    x_interp = np.linspace(t_start + binsize, t_end, n_bins)
    f = interp1d(bt, bv, kind='linear', fill_value='extrapolate')
    binned_beh[trial_idx] = f(x_interp).astype(np.float32)
```

iii. The AI states this matches the reference code's `get_behavior_per_interval` function.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 bins using global tercile thresholds computed across all sessions. The percentile edges are computed at 33.3% and 66.7% of the pooled data. Values are then assigned to bins 0, 1, or 2 using `np.digitize`.

ii.
```python
# Global tercile edges:
all_wheel_flat = np.concatenate([w.flatten() for w in all_wheel_speed_raw])
wheel_edges = np.percentile(all_wheel_flat[~np.isnan(all_wheel_flat)], [100/3, 200/3])
# Discretization:
wheel_disc = np.digitize(sm['wheel_binned'], wheel_edges).astype(np.int64)
```

iii. The AI states: "3 bins using global tercile thresholds" and notes this produces balanced distributions (~33.3% per bin).

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated to the same bin centers as the neural data using the same `np.linspace(t_start + binsize, t_end, n_bins)` formula. This ensures temporal alignment between neural activity and wheel speed.

ii.
```python
wheel_binned, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, WINDOW, BINSIZE
)
```

iii. The AI states it matches the reference code's bin center computation.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from the `whiskerMotionEnergy` column of the left camera motion energy data (`SessionLoader.motion_energy['leftCamera']`). If left camera data is unavailable, right camera data is used as fallback.

ii.
```python
sl.load_motion_energy(views=['left'])
me_times = sl.motion_energy['leftCamera']['times'].values
me_values = sl.motion_energy['leftCamera']['whiskerMotionEnergy'].values
```

iii. The AI states: "Reference code tries left first: load_target_behavior(one, eid, 'left-whisker-motion-energy'), falls back to right if unavailable."

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Whisker motion energy is linearly interpolated to bin centers within the trial window, using the same interpolation method as wheel speed.

ii.
```python
whisker_binned, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, WINDOW, BINSIZE
)
```

iii. The AI states this follows the same interpolation approach as the reference code.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Whisker ME is discretized into 3 bins using global tercile thresholds, same approach as wheel speed.

ii.
```python
all_whisker_flat = np.concatenate([w.flatten() for w in all_whisker_me_raw])
whisker_edges = np.percentile(all_whisker_flat[~np.isnan(all_whisker_flat)], [100/3, 200/3])
whisker_disc = np.digitize(sm['whisker_binned'], whisker_edges).astype(np.int64)
```

iii. The AI notes both wheel speed and whisker ME use the same global tercile discretization approach.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker ME is interpolated to the same bin centers as neural data, using the same formula `np.linspace(t_start + binsize, t_end, n_bins)`.

ii.
```python
whisker_binned, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, WINDOW, BINSIZE
)
```

iii. Same alignment approach as wheel speed and neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is handled at multiple levels: (1) Sessions where no probes could be loaded are skipped. (2) Sessions with fewer than 10 valid trials are skipped. (3) Sessions without whisker motion energy data are skipped (15 sessions). (4) Individual trials where behavioral data (wheel or whisker) has insufficient temporal coverage, NaN values, or too few data points are excluded via the `combined_mask`. (5) NaN alignment times cause trials to be skipped in the binning functions.

ii.
```python
# Skip sessions with too few trials:
if mask.sum() < 10:
    print(f'  Skipping: only {mask.sum()} valid trials')
    continue
# Skip sessions without whisker ME:
if whisker_binned is None:
    print(f'  Skipping: no whisker motion energy')
    continue
# Combined behavioral mask:
combined_mask = wheel_mask & whisker_mask
if combined_mask.sum() < 10:
    print(f'  Skipping: only {combined_mask.sum()} trials with valid behavior')
    continue
# Individual trial checks in interpolation:
if len(bt) < 2:
    good_mask[trial_idx] = False
if np.any(np.isnan(bv)):
    good_mask[trial_idx] = False
```

iii. The AI reports only 4 trials out of 190,151 had all-zero neural data, and all validation checks passed.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading spike sorting data per probe via `SpikeSortingLoader` (involves disk I/O and potentially network downloads). (2) Binning spikes per trial in `bin_spikes_in_window` which iterates over all trials with searchsorted operations. (3) Interpolating behavioral data per trial in `interpolate_behavior_to_bins`.

ii.
```python
# Spike binning loops over every trial:
for trial_idx in range(n_trials):
    t_start = align_times[trial_idx] + window[0]
    # ...searchsorted, binning...
# Behavior interpolation also loops:
for trial_idx in range(n_trials):
    # ...interpolation per trial...
```

iii. The AI uses vectorized numpy operations within each trial (searchsorted, np.add.at) but still loops over trials sequentially.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loops in `bin_spikes_in_window` and `interpolate_behavior_to_bins` are the main candidates. The spike binning loop could potentially use fully vectorized approaches (e.g., broadcasting with trial-offset spike times). The behavior interpolation is harder to vectorize due to varying data availability per trial. The final output construction loop (iterating over trials to build per-trial arrays) is also a candidate.

ii.
```python
# This loop iterates per trial:
for trial_idx in range(n_trials):
    # searchsorted + binning per trial

# Output construction loop:
for trial in range(n_trials):
    session_neural.append(sm['binned_spikes'][trial].astype(np.float32))
    # ...build input/output arrays...
```

iii. The reference code uses multiprocessing via `multiprocessing.Pool` to parallelize spike binning and behavior interpolation, while the AI uses sequential loops. The reference code's `get_spike_data_per_interval` uses `p.imap_unordered` for parallel processing.

## 12-c. What processing does the code repeat multiple times?

i. The AI loads the `SessionLoader` for trials and then reuses it for wheel and whisker ME loading, which is efficient. However, the behavior interpolation function is called separately for wheel and whisker ME with nearly identical logic. The `interpolate_behavior_to_bins` function is called twice with the same window/binsize parameters but different data.

ii.
```python
wheel_binned, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, WINDOW, BINSIZE
)
# ... then separately ...
whisker_binned, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, WINDOW, BINSIZE
)
```

iii. This duplication is minor and standard practice. More significant is that the trial window computation (t_start, t_end per trial) is computed independently in both spike binning and behavior interpolation.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI replicates per-trial variables (choice, prior) across all 100 time bins to create time-varying output arrays of shape (4, 100). This is unnecessary because choice and prior are constant within a trial - the downstream decoder only needs a single per-trial value. This increases memory usage by 100x for these variables. Additionally, the AI includes all clusters (even completely silent neurons that never spike), which adds zero-valued rows to the neural matrix.

ii.
```python
# Replicating per-trial values across time bins (unnecessary):
choice_arr = np.full(N_BINS, sm['choice'][trial], dtype=np.int64)
prior_arr = np.full(N_BINS, sm['prior'][trial], dtype=np.int64)
out = np.stack([choice_arr, prior_arr, wheel_arr, whisker_arr], axis=0)  # (4, 100)
```

iii. The AI justifies this by stating: "Making all time-varying enables the decoder to handle them uniformly." While this may simplify the decoder interface, it wastes memory and compute for per-trial variables.
