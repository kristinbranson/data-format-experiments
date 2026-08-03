# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data using the ONE API connected to the IBL OpenAlyx database. It reads a `bwm_release.csv` file to get a manifest of all probe insertions (699 probes across 459 sessions), then iterates over unique session eids. For each session, it uses `SpikeSortingLoader` to load spike sorting data for each probe insertion, `SessionLoader` to load trial information, wheel data, and whisker motion energy data.

ii.
```python
one = ONE(
    base_url='https://openalyx.internationalbrainlab.org',
    password='international', silent=True,
    cache_dir=args.cache_dir
)

bwm_df = pd.read_csv('/app/code/code_zhang2025/data/bwm_release.csv', index_col=0)

# Iterates over unique eids
eids = bwm_df['eid'].unique()
for eid_idx, eid in enumerate(eids):
    rows = bwm_df[bwm_df['eid'] == eid]
    # ... load spike sorting, trials, wheel, whisker ME
```

iii. The AI's justification (from CONVERSION_NOTES.md) states this follows the reference code's `prepare_data` function which uses ONE API and the BWM release CSV. The reference code's `0_data_caching.py` similarly uses `ONE()` and `bwm_release.csv`.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `bwm_df` DataFrame's `subject` column. Each unique subject name is tracked in a `subject_to_idx` dictionary that maps names to integer indices. The `subjects` list and `subject_idx` array are built up as sessions are processed.

ii.
```python
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects_list)
    subjects_list.append(subject)
subj_idx = subject_to_idx[subject]
```

iii. The AI assigns subjects based on the BWM DataFrame metadata. This matches the reference code approach where subject information comes from the same `bwm_df`.

## 1-c. How are the data split into sessions?

i. Each unique `eid` in the BWM release DataFrame corresponds to one session. The AI iterates over all unique eids and processes each independently. Multiple probes within the same session are merged (see probe merging below). The final data has one entry per session in the `neural`, `input`, and `output` lists.

ii.
```python
eids = bwm_df['eid'].unique()
for eid_idx, eid in enumerate(eids):
    rows = bwm_df[bwm_df['eid'] == eid]
    # Process session...
    neural_list.append(session_neural)
```

iii. This matches the reference code's approach in `0_data_caching.py` which also iterates over eids.

## 1-d. How are the data split into trials?

i. Within each session, trial information is loaded via `SessionLoader.load_trials()`. Each trial in the trials DataFrame has associated timing information (stimOn_times, feedback_times, etc.), choice, and probabilityLeft. After filtering (see 1-e), valid trials are extracted with their timing used to bin spikes and behavior into per-trial arrays.

ii.
```python
sl = SessionLoader(one=one, eid=eid)
sl.load_trials()
trials = sl.trials
# ... apply mask ...
masked_trials = trials[mask].reset_index(drop=True)
align_times = masked_trials['stimOn_times'].values
```

iii. Each trial is defined by its stimulus onset time and a 2s window (-0.5 to 1.5s), matching the reference code's trial segmentation approach.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using three criteria: (1) Reaction time (firstMovement_times - stimOn_times) must be between 0.08s and 2.0s; (2) Key event columns cannot contain NaN values: stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType; (3) No-choice trials (choice == 0) are excluded. Additionally, trials where wheel or whisker motion energy interpolation fails are excluded via a combined behavioral mask.

ii.
```python
def load_trials_and_mask(one, eid, min_rt=0.08, max_rt=2.0):
    nan_exclude = [
        'stimOn_times', 'choice', 'feedback_times',
        'probabilityLeft', 'firstMovement_times', 'feedbackType'
    ]
    mask = pd.Series(True, index=trials.index)
    rt = trials['firstMovement_times'] - trials['stimOn_times']
    if min_rt is not None:
        mask &= rt >= min_rt
    if max_rt is not None:
        mask &= rt <= max_rt
    for event in nan_exclude:
        mask &= ~trials[event].isnull()
    mask &= trials['choice'] != 0
    return trials, mask, sl
```

iii. The AI states this matches `load_trials_and_mask` defaults in the reference code. The reference code uses the same defaults (min_rt=0.08, max_rt=2.0, default nan_exclude list, exclude_nochoice=True). However, the reference code's `prepare_data` also passes `max_trial_len=10.0` to `load_trials_and_mask`, which the AI omits.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times (`spikes['times']`) and cluster assignments (`spikes['clusters']`) loaded via `SpikeSortingLoader`. These are raw spike-sorted data from Neuropixels probes.

ii.
```python
ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
spikes, clusters, channels = ssl.load_spike_sorting()
clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
return spikes, clusters_labeled
```

iii. The AI uses the same loading mechanism as the reference code's `load_spiking_data` function.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 20ms non-overlapping time bins within a [-0.5, 1.5]s window around stimulus onset, producing spike count matrices of shape (n_neurons, 100) per trial. The AI implements custom spike binning using searchsorted and np.add.at for accumulation. Multiple probes within a session are merged by concatenating spikes and re-indexing clusters.

ii.
```python
def bin_spikes_in_window(spike_times, spike_clusters, cluster_ids,
                          align_times, window, binsize):
    n_bins = int(np.round((window[1] - window[0]) / binsize))  # = 100
    # ...
    bin_indices = np.floor((times_in_window - t_start) / binsize).astype(np.int32)
    np.clip(bin_indices, 0, n_bins - 1, out=bin_indices)
    np.add.at(binned[trial_idx], (cluster_indices[valid], bin_indices[valid]), 1)
```

iii. The reference code uses `bincount2D` from `iblutil.numerical` for spike binning, while the AI reimplements this with custom numpy operations. Both produce spike count matrices. The reference uses `np.ceil` for n_bins while the AI uses `np.round`; for the specific values (2.0/0.02), both yield 100.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No quality control filtering is applied to neurons. All spike-sorted clusters are included (no QC threshold). The AI uses `qc=None` (the default), which returns all clusters regardless of quality label.

ii.
```python
def load_spiking_data(one, pid, eid='', pname=''):
    """Uses all clusters (no QC filter)"""
    ssl = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters_labeled = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
    return spikes, clusters_labeled
```

iii. The AI's CONVERSION_NOTES.md states: "The reference code's `prepare_data` function calls `load_spiking_data` without the `qc` parameter, defaulting to `qc=None` which returns all clusters." This matches the reference code where `prepare_data` calls `load_spiking_data(one, pid, eid=eid, pname=probe_name)` without a `qc` argument.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, spikes are binned within the window [stimOn_times - 0.5s, stimOn_times + 1.5s], producing 100 time bins of 20ms each.

ii.
```python
WINDOW = (-0.5, 1.5)
align_times = masked_trials['stimOn_times'].values
binned_spikes = bin_spikes_in_window(
    spikes['times'], spikes['clusters'], cluster_ids,
    align_times, WINDOW, BINSIZE
)
```

iii. The AI states this matches the reference code's `params = {'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}` and the methods paper's description: "For choice, we align trials to the stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20ms (0.02s). No temporal rebinning is applied beyond the initial spike binning. This produces T=100 time steps per 2-second trial.

ii.
```python
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.round((WINDOW[1] - WINDOW[0]) / BINSIZE))  # = 100
```

iii. The AI notes the reference code uses `binsize=0.02` and the methods paper states "divided into 20-ms bins, producing T = 100 time steps." The AI notes the methods paper mentions 50ms bins for choice/prior decoding, but follows the reference code's 20ms universal binsize.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Time since stimulus onset is not derived from any specific raw data variable. It is computed as a fixed array of bin center times relative to the alignment window, independent of the specific stimulus onset time of each trial.

ii.
```python
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)
# Result: array from -0.48 to 1.5, 100 bins
```

iii. This is the same for every trial since the window is always [-0.5, 1.5]s relative to stimulus onset. The bin centers match the reference code's interpolation grid: `np.linspace(interval_beg + binsize, interval_end, n_bins)`.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A fixed linspace array is created from -0.48s to 1.5s with 100 points, representing bin center times relative to stimulus onset. This is replicated identically for every trial.

ii.
```python
time_since_stim = np.linspace(
    WINDOW[0] + BINSIZE, WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The values represent time since stimulus onset at each bin center, consistent with the reference code's bin center computation.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time since stimulus onset array has the same number of time bins (100) as the neural data, since both use the same window and bin size. Each element of the time array corresponds to the same time bin in the neural data. The input is stored as shape (2, T) with time_since_stim as the first row.

ii.
```python
inp = np.stack([time_input, trial_num], axis=0)  # (2, T)
session_input.append(inp.astype(np.float32))
```

iii. Alignment is implicit since both neural and time arrays share the same binning scheme.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from the `probabilityLeft` column in the trials DataFrame. Block boundaries are detected when `probabilityLeft` changes from one trial to the next.

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

iii. The AI determines block boundaries by detecting changes in `probabilityLeft`, which defines the prior probability blocks in the IBL task. This is a reasonable approach since the reference code doesn't explicitly compute trial number in block (it's a decoder-specific input).

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Starting from the first trial, a counter is initialized to 1. When `probabilityLeft` changes, the counter resets to 1. Otherwise, it increments by 1 for each subsequent trial. The counter is computed on ALL trials (including those that will be masked), then only the masked values are extracted. The result is replicated across all time bins for each trial.

ii.
```python
trial_num_in_block = compute_trial_number_in_block(trials, mask.values)
trial_num_in_block = trial_num_in_block[combined_mask]
# Per trial, replicated across time:
trial_num = np.full(N_BINS, sm['trial_num_in_block'][trial], dtype=np.float32)
```

iii. The block counter runs over ALL trials (not just valid ones), ensuring that block boundaries are correctly identified even when some trials within a block are filtered out.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column in the IBL trials DataFrame, where choice=-1 represents left and choice=1 represents right.

ii.
```python
choice_vals = masked_trials.iloc[masked_idx]['choice'].values.copy()
choice = np.where(choice_vals == 1, 1, 0).astype(np.int64)
```

iii. The AI remaps IBL's {-1, 1} encoding to the decoder spec's {0, 1} where left=0, right=1.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The raw IBL choice values (-1 for left, 1 for right) are mapped to binary values: left (-1) becomes 0, right (1) becomes 1. No-choice trials (choice=0) are already excluded by the trial mask. The per-trial choice value is replicated across all 100 time bins.

ii.
```python
choice = np.where(choice_vals == 1, 1, 0).astype(np.int64)
# Then replicated across time:
choice_arr = np.full(N_BINS, sm['choice'][trial], dtype=np.int64)
```

iii. This matches the decoder task specification: "left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from the `probabilityLeft` column in the trials DataFrame, which takes values 0.2, 0.5, or 0.8.

ii.
```python
pLeft = masked_trials.iloc[masked_idx]['probabilityLeft'].values
prior = np.zeros(len(pLeft), dtype=np.int64)
prior[pLeft == 0.2] = 0
prior[pLeft == 0.5] = 1
prior[pLeft == 0.8] = 2
```

iii. This directly maps the three possible prior probability values to categorical indices as specified in the decoder task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The raw `probabilityLeft` values are mapped to categorical indices: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. The per-trial value is replicated across all 100 time bins.

ii.
```python
prior[pLeft == 0.2] = 0
prior[pLeft == 0.5] = 1
prior[pLeft == 0.8] = 2
# Then replicated:
prior_arr = np.full(N_BINS, sm['prior'][trial], dtype=np.int64)
```

iii. This matches the decoder task specification: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from the wheel velocity data loaded via `SessionLoader.load_wheel()`. Specifically, it uses `sess_loader.wheel['velocity']` and takes its absolute value.

ii.
```python
sl.load_wheel()
wheel_times = sl.wheel['times'].values
wheel_speed = np.abs(sl.wheel['velocity'].values)
```

iii. The AI states this matches the reference code's `load_target_behavior` for 'wheel-speed': `np.abs(sess_loader.wheel['velocity'].to_numpy())`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The absolute value of wheel velocity is taken to get speed. This continuous signal is then interpolated to 20ms bin centers within the trial window using linear interpolation. The interpolated values are later discretized into 3 categories.

ii.
```python
wheel_speed = np.abs(sl.wheel['velocity'].values)
wheel_binned, wheel_mask = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, align_times, WINDOW, BINSIZE
)
```

iii. Matches the reference code's wheel-speed loading and the interpolation approach in `get_behavior_per_interval`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 bins using global tercile thresholds computed across ALL sessions. The 33rd and 67th percentiles of all wheel speed values are used as bin edges. `np.digitize` then assigns each value to bin 0, 1, or 2.

ii.
```python
all_wheel_flat = np.concatenate([w.flatten() for w in all_wheel_speed_raw])
wheel_edges = np.percentile(all_wheel_flat[~np.isnan(all_wheel_flat)], [100/3, 200/3])
# Per session:
wheel_disc = np.digitize(sm['wheel_binned'], wheel_edges).astype(np.int64)
```

iii. The AI uses global (cross-session) tercile thresholds, producing balanced ~33.3% bins. The instructions only specify "discretized into 3 bins" without specifying the method, so this is a reasonable choice.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated to the same bin centers as the neural data using `np.linspace(t_start + binsize, t_end, n_bins)`, producing 100 values per trial matching the 100 time bins of neural data.

ii.
```python
def interpolate_behavior_to_bins(beh_times, beh_values, align_times, window, binsize):
    n_bins = int(np.round((window[1] - window[0]) / binsize))
    x_interp = np.linspace(t_start + binsize, t_end, n_bins)
    f = interp1d(bt, bv, kind='linear', fill_value='extrapolate')
    binned_beh[trial_idx] = f(x_interp).astype(np.float32)
```

iii. The bin centers match the reference code's interpolation: `np.linspace(interval_begs[interval_idx] + binsize, interval_ends[interval_idx], n_bins)`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from the left camera's `whiskerMotionEnergy` column loaded via `SessionLoader.load_motion_energy(views=['left'])`. If left camera data is unavailable, it falls back to the right camera.

ii.
```python
sl.load_motion_energy(views=['left'])
me_times = sl.motion_energy['leftCamera']['times'].values
me_values = sl.motion_energy['leftCamera']['whiskerMotionEnergy'].values
```

iii. The AI states this matches the reference code which tries left first then falls back to right: `load_target_behavior(one, eid, 'left-whisker-motion-energy')`.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw whisker motion energy time series is interpolated to 20ms bin centers within the trial window using linear interpolation, then later discretized into 3 categories.

ii.
```python
whisker_binned, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, WINDOW, BINSIZE
)
```

iii. Same interpolation approach as wheel speed, matching the reference code's `get_behavior_per_interval`.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Whisker motion energy is discretized into 3 bins using global tercile thresholds computed across ALL sessions, using the same approach as wheel speed.

ii.
```python
all_whisker_flat = np.concatenate([w.flatten() for w in all_whisker_me_raw])
whisker_edges = np.percentile(all_whisker_flat[~np.isnan(all_whisker_flat)], [100/3, 200/3])
whisker_disc = np.digitize(sm['whisker_binned'], whisker_edges).astype(np.int64)
```

iii. Same global tercile approach as wheel speed. Instructions specify "discretized into 3 bins" without specifying the method.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned identically to wheel speed: interpolated to the same 100 bin centers as neural data using `np.linspace(t_start + binsize, t_end, n_bins)`.

ii.
```python
whisker_binned, whisker_mask = interpolate_behavior_to_bins(
    me_times, me_values, align_times, WINDOW, BINSIZE
)
```

iii. Same alignment as wheel speed and neural data, matching the reference code.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies handle missing/problematic data: (1) If a probe fails to load, it's skipped with a warning. (2) If no probes load for a session, the session is skipped. (3) Sessions with fewer than 10 valid trials are skipped. (4) If whisker motion energy is unavailable for both cameras, the session is skipped. (5) Trials where behavioral interpolation fails (insufficient data points, NaN values, poor temporal coverage) are marked as bad via the behavior mask. (6) Sessions with fewer than 10 valid trials after behavior masking are skipped. (7) General exception handling catches and logs unexpected errors per session.

ii.
```python
# Probe loading failure
except Exception as e:
    print(f'  Warning: Failed to load probe {pid}: {e}')

# Session with too few trials
if mask.sum() < 10:
    print(f'  Skipping: only {mask.sum()} valid trials')
    skipped_sessions += 1
    continue

# Behavior interpolation checks
if len(bt) < 2:
    good_mask[trial_idx] = False
if np.any(np.isnan(bv)):
    good_mask[trial_idx] = False
if np.abs(t_start - bt[0]) > binsize:
    good_mask[trial_idx] = False
```

iii. The AI skipped 15 sessions due to missing whisker motion energy data, resulting in 444 sessions (vs 459 total in BWM release). The approach is conservative: rather than imputing missing data, problematic trials or sessions are excluded.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading spike sorting data via SpikeSortingLoader (involves downloading/reading large binary files from disk/cache). (2) Spike binning in `bin_spikes_in_window`, which loops over all trials and performs searchsorted + accumulation for each. (3) Behavioral interpolation in `interpolate_behavior_to_bins`, which loops over trials. (4) The overall conversion processes 459 sessions sequentially.

ii.
```python
# Spike binning loop (most compute-intensive)
for trial_idx in range(n_trials):
    i_start = np.searchsorted(sorted_times, t_start, side='left')
    i_end = np.searchsorted(sorted_times, t_end, side='left')
    # ... binning logic per trial
```

iii. The full conversion required processing all sessions sequentially, making spike sorting loading and binning the dominant costs.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-level spike binning loop in `bin_spikes_in_window` iterates over each trial individually. This could potentially be vectorized using `bincount2D` (as the reference code does) which handles multiple intervals at once. The behavior interpolation loop in `interpolate_behavior_to_bins` also iterates per trial, though the reference code parallelizes this with `multiprocessing.Pool`.

ii.
```python
# Trial-level loop that could be vectorized
for trial_idx in range(n_trials):
    t_start = align_times[trial_idx] + window[0]
    t_end = align_times[trial_idx] + window[1]
    # ... per-trial binning

# Behavior interpolation loop
for trial_idx in range(n_trials):
    # ... per-trial interpolation
```

iii. The reference code uses multiprocessing for both spike binning and behavior interpolation. The AI's approach is single-threaded but uses efficient numpy operations.

## 10-c. What processing does the code repeat multiple times?

i. The code creates a new `SessionLoader` for both trial loading and wheel/whisker data loading, though it reuses the `sl` object for wheel loading. The `interpolate_behavior_to_bins` function is called twice (once for wheel, once for whisker) with the same trial alignment times, recomputing `t_start` and `t_end` for each trial redundantly. The `compute_trial_number_in_block` iterates over all trials even though only masked trials are needed.

ii.
```python
# SessionLoader reused for wheel but recreated would happen if separate calls
trials, mask, sl = load_trials_and_mask(one, eid)
sl.load_wheel()  # reuses sl
# But whisker ME is loaded on same sl:
sl.load_motion_energy(views=['left'])
```

iii. The AI avoids creating separate SessionLoader instances by reusing `sl` from trial loading for wheel and whisker data.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Per-trial choice and prior values are replicated across all 100 time bins (`np.full(N_BINS, ...)`), creating time-varying outputs for per-trial variables. The downstream decoder likely treats these as per-trial values, making the replication wasteful in terms of storage. (2) The AI stores all clusters including those in "void" or "root" brain regions, which may not be useful for neuroscientific analysis. (3) The code computes and stores the `discretize_to_bins` function (lines 244-265) but doesn't actually use it - instead using `np.digitize` with manually computed edges for discretization.

ii.
```python
# Replicating per-trial values across time
choice_arr = np.full(N_BINS, sm['choice'][trial], dtype=np.int64)
prior_arr = np.full(N_BINS, sm['prior'][trial], dtype=np.int64)
out = np.stack([choice_arr, prior_arr, wheel_arr, whisker_arr], axis=0)

# Unused function
def discretize_to_bins(values, n_bins=3):
    # ... never called
```

iii. The time-replication of per-trial outputs follows the instructions which say "If at all possible, make it time-varying." The unused `discretize_to_bins` function is dead code.
