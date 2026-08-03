# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by reading a CSV file (`bwm_release.csv`) that lists all probe insertions (PIDs) with their session IDs, subjects, labs, dates, and probe names. It groups entries by session (eid), then for each session constructs the file path from the lab/subject/date fields and loads files directly from the ONE cache directory structure. It does NOT use the ONE API or `SessionLoader`/`SpikeSortingLoader`.

ii.
```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
sessions = {}
for _, row in bwm_df.iterrows():
    eid = row.eid
    if eid not in sessions:
        sessions[eid] = []
    sessions[eid].append({
        'pid': row.pid,
        'probe_name': row.probe_name,
        'subject': row.subject,
        'lab': row.lab,
        'date': row.date,
    })
```

```python
def find_session_path(lab, subject, date):
    for sess_num in ['001', '002', '003']:
        p = BASE_PATH / lab / 'Subjects' / subject / date / sess_num / 'alf'
        if p.exists():
            return p
    return None
```

iii. The AI chose to load data directly from files rather than using the ONE API, reasoning that direct file loading avoids the dependency on `SpikeSortingLoader`. The session list comes from the `bwm_release.csv` that ships with the reference code.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column of the `bwm_release.csv`. After processing, unique subjects are collected and sorted, and a `subject_to_idx` mapping is built.

ii.
```python
all_subjects = sorted(list(set(sess['subject'] for sess in session_results)))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
```

iii. Subject identity is taken directly from the CSV metadata, no parsing of paths required.

## 1-c. How are the data split into sessions?

i. Sessions are identified by the unique `eid` column in `bwm_release.csv`. Each unique eid corresponds to one session, and probes within the same session are grouped together.

ii.
```python
sessions = {}
for _, row in bwm_df.iterrows():
    eid = row.eid
    if eid not in sessions:
        sessions[eid] = []
    sessions[eid].append({...})
```

iii. The eid is the session identifier from the BWM release CSV. Sessions are already the natural unit.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) has one row per trial. The AI loads it with `pd.read_parquet()` and each row becomes a trial.

ii.
```python
trials_df = pd.read_parquet(trials_file)
```

iii. The trials table is already organized as one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several filters: reaction time between 0.08s and 2.0s, trial length (feedback_times - goCue_times) <= 10s, NaN exclusion on multiple fields (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType), and exclusion of no-choice trials (choice != 0). Additionally, trials where behavior interpolation fails (insufficient wheel or whisker coverage) are dropped via a `combined_valid` mask.

ii.
```python
rt = trials_df['firstMovement_times'] - trials_df['stimOn_times']
mask &= (rt >= MIN_RT)
mask &= (rt <= MAX_RT)

if MAX_TRIAL_LEN is not None:
    if 'goCue_times' in trials_df.columns:
        trial_len = trials_df['feedback_times'] - trials_df['goCue_times']
        mask &= (trial_len <= MAX_TRIAL_LEN)

for event in NAN_EXCLUDE:
    if event in trials_df.columns:
        mask &= ~trials_df[event].isna()

if EXCLUDE_NOCHOICE:
    mask &= (trials_df['choice'] != 0)
```

```python
combined_valid = wheel_valid & me_valid
```

iii. The AI identified these filters from the reference code's `load_trials_and_mask` function parameters. The NaN exclusion on multiple fields (including feedback_times and feedbackType) and the max_trial_len filter go beyond what the reference solution applies.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spikes.times.npy` and `spikes.clusters.npy` from each probe's pykilosort directory. Also `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` for brain region mapping.

ii.
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
cluster_channels = np.load(spike_dir / 'clusters.channels.npy').flatten()
chan_brain_ids = np.load(spike_dir / 'channels.brainLocationIds_ccf_2017.npy').flatten()
```

iii. These are the fundamental spike sorting outputs needed for binning and region assignment.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 20ms bins over a 2s window (-0.5 to 1.5s from stimulus onset), giving 100 bins per trial. Spike counts are stored as float32 but NOT divided by the bin width (i.e., they remain as raw spike counts, not firing rates). When a session has multiple probes, the clusters are merged by offsetting cluster indices and sorting by time.

ii.
```python
binned = np.zeros((n_trials, n_clusters_total, n_bins), dtype=np.float32)
...
bin_idx = np.minimum(np.floor((trial_times - t_start) / binsize).astype(int), n_bins - 1)
np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)
```

iii. The AI bins spikes into counts but does not convert to firing rates (Hz). The binned values remain as spike counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. NO quality control filtering is applied. All clusters from the spike sorting are retained, regardless of their quality label.

ii. There is no QC filtering code in the AI's script. The `load_spike_data` function loads all clusters without checking any quality metrics:
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
```

iii. The AI noted that `prepare_data` in the reference code calls `load_spiking_data` with `qc=None` and concluded that no QC filtering should be applied. From CONVERSION_NOTES.md: "No QC filtering: `load_spiking_data` called with `qc=None` in `prepare_data`" and "Neuron curation: No QC filtering (all clusters used)".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to stimulus onset (`stimOn_times`). The binning window starts at `stim_on + TIME_WINDOW[0]` (-0.5s) and ends at `stim_on + TIME_WINDOW[1]` (1.5s).

ii.
```python
stim_on = valid_trials[ALIGN_TIME].values
interval_starts = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
```

```python
t_start = interval_starts[trial_idx]
idx_start = np.searchsorted(spike_times, t_start, side='left')
idx_end = np.searchsorted(spike_times, t_end, side='left')
```

iii. Alignment to stimulus onset matches the reference code's configuration.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20ms bins (BINSIZE = 0.02), giving 100 bins for the 2s window. No rebinning is applied.

ii.
```python
BINSIZE = 0.02  # 20ms bins
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))  # 100 bins
```

iii. This matches the reference code's binsize parameter of 0.02s.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is a synthetic variable derived from the time window parameters. It is NOT derived from `stimOn_times` per se, but is a fixed grid representing time within the trial window.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The AI constructs the time grid using `np.linspace(-0.48, 1.5, 100)`, giving values from -0.48 to 1.50. This differs from the reference solution which uses bin centers at -0.49 to 1.49.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time values are computed as `np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)` = `np.linspace(-0.48, 1.5, 100)`. This produces 100 evenly spaced values from -0.48 to 1.50, which correspond to the RIGHT edges of the neural bins rather than the bin centers.

ii.
```python
time_since_stim = np.linspace(
    TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS
).astype(np.float32)
```

iii. The AI adopted this formula from the reference code's `get_behavior_per_interval` function which uses `np.linspace(interval_beg + binsize, interval_end, n_bins)`. However, the reference solution uses proper bin centers: `EDGES[:-1] + BIN / 2` = [-0.49, -0.47, ..., 1.49], which are shifted by half a bin (0.01s) from the AI's values.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input uses the same `np.linspace` grid as the behavior interpolation, but this grid does not exactly match the neural bin centers. The neural bins have edges at [-0.5, -0.48, ..., 1.48, 1.5] with centers at [-0.49, -0.47, ..., 1.49], while the input time values are [-0.48, -0.46, ..., 1.50], shifted by half a bin.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

iii. The AI uses the same time grid for both the time input and behavior interpolation, ensuring internal consistency within the AI's code, but this grid is shifted by 0.01s relative to the actual bin centers used for spike counting.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A block boundary is detected when `probabilityLeft` changes value.

ii.
```python
def compute_trial_number_in_block(prob_left):
    trial_nums = np.zeros(len(prob_left), dtype=np.float32)
    current_block = prob_left.iloc[0] if hasattr(prob_left, 'iloc') else prob_left[0]
    count = 0
    for i in range(len(prob_left)):
        val = prob_left.iloc[i] if hasattr(prob_left, 'iloc') else prob_left[i]
        if val == current_block:
            count += 1
        else:
            current_block = val
            count = 1
        trial_nums[i] = count
    return trial_nums
```

iii. The blocks are recovered from changes in `probabilityLeft`, which is held constant within a block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Two issues: (1) The trial count starts at 1 rather than 0. (2) The block boundaries are computed on the FILTERED trials rather than on all trials before filtering.

ii. The function is called on filtered data:
```python
trial_num_in_block = compute_trial_number_in_block(
    valid_trials_final['probabilityLeft']
)
```

Where `valid_trials_final` is the already-filtered trial set. The count starts at 1:
```python
count = 0
...
if val == current_block:
    count += 1
else:
    current_block = val
    count = 1
```

iii. The reference solution computes trial_in_block on ALL trials BEFORE filtering (using `trials.groupby(block).cumcount()` which starts at 0), then selects the filtered subset. The AI's approach changes block numbering when filtered trials create artificial block boundaries.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, which takes values +1 (left), -1 (right), and 0 (no response).

ii.
```python
choice = valid_trials_final['choice'].values.copy()
choice_binary = np.where(choice == -1, 0, 1).astype(np.float32)
```

iii. The choice column is directly available in the trials table.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps choice=-1 to 0 and choice=+1 to 1. In IBL convention, +1 is a leftward choice and -1 is a rightward choice. So the AI maps: LEFT(+1) -> 1, RIGHT(-1) -> 0. This is the OPPOSITE of the instructions and reference, which specify left=0, right=1.

ii.
```python
choice_binary = np.where(choice == -1, 0, 1).astype(np.float32)
```

The output_values say `['left', 'right']` (0=left, 1=right), but the code maps left(+1)->1 and right(-1)->0.

iii. The AI's code comment says "left(-1)->0, right(1)->1" which incorrectly assumes IBL's convention is -1=left, +1=right. Actually IBL uses +1=left, -1=right. So the mapping is inverted compared to the instructions ("left = 0, right = 1").

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, and 0.8.

ii.
```python
prob_left = valid_trials_final['probabilityLeft'].values.copy()
prior_cat = np.zeros(len(prob_left), dtype=np.float32)
prior_cat[prob_left == 0.2] = 0
prior_cat[prob_left == 0.5] = 1
prior_cat[prob_left == 0.8] = 2
```

iii. The mapping follows the instructions: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct categorical mapping of the three probability values to integers 0, 1, 2.

ii.
```python
prior_cat[prob_left == 0.2] = 0
prior_cat[prob_left == 0.5] = 1
prior_cat[prob_left == 0.8] = 2
```

iii. Straightforward mapping as specified in the instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`, which provide the raw wheel position and its timestamps.

ii.
```python
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The wheel position and timestamps are the raw inputs for computing speed.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Three steps matching `SessionLoader.load_wheel()`: (1) Interpolate position to uniform 1000Hz sampling, (2) Apply 8th-order Butterworth low-pass filter at 20Hz corner frequency and differentiate to get velocity, (3) Take absolute value for speed. The speed is then interpolated to match the neural time bins using `interp1d` with `np.linspace(t_start + binsize, t_end, n_bins)`.

ii.
```python
t_uniform = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
pos_interp = interp1d(timestamps, position, kind='linear')(t_uniform)
sos = scipy.signal.butter(N=order, Wn=corner_frequency / fs * 2, btype='lowpass', output='sos')
vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
speed = np.abs(vel).astype(np.float32)
```

iii. The AI reimplemented the wheel processing to match `interpolate_position` and `velocity_filtered` from the IBL library. The reference solution delegates this to `SessionLoader.load_wheel()`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Discretized into 3 bins using GLOBAL quantile boundaries computed across ALL sessions. The 1/3 and 2/3 quantiles are computed on the concatenated wheel speed values from all sessions, then applied uniformly.

ii.
```python
all_wheel = np.concatenate(all_wheel)
wheel_quantiles = np.array([-np.inf,
                             np.quantile(all_wheel, 1/3),
                             np.quantile(all_wheel, 2/3),
                             np.inf])
wheel_disc = np.digitize(sess['wheel_binned'], wheel_quantiles[1:-1]).astype(np.float32)
```

iii. The AI chose global discretization, while the reference solution discretizes PER SESSION using session-specific percentiles. Global discretization means the three categories are NOT equally sized within each session.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated to time points defined by `np.linspace(t_start + binsize, t_end, n_bins)`, which is the same grid used for the time input. This grid is shifted by half a bin (0.01s) relative to the neural bin centers.

ii.
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
f_interp = interp1d(seg_times, seg_vals, kind='linear', fill_value='extrapolate')
result[trial_idx] = f_interp(x_interp)
```

iii. The AI uses scipy's `interp1d` with extrapolation enabled, while the reference uses `np.interp` at bin centers. The time grids differ by 0.01s.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` (preferred) or `rightCamera.ROIMotionEnergy.npy`, with corresponding `_ibl_leftCamera.times.npy` or `_ibl_rightCamera.times.npy`.

ii.
```python
me_file = find_versioned_file(sess_path, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_versioned_file(sess_path, '_ibl_leftCamera.times.npy')
if me_file is None or times_file is None:
    me_file = find_versioned_file(sess_path, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_versioned_file(sess_path, '_ibl_rightCamera.times.npy')
```

iii. Left camera is preferred over right, matching the reference approach.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are loaded and used directly (no additional filtering or normalization). NaN values are removed. The trace is interpolated to the neural time bins using the same `interpolate_behavior` function as the wheel.

ii.
```python
me_values = np.load(me_file).flatten()
me_times = np.load(times_file).flatten()
valid = ~np.isnan(me_values) & ~np.isnan(me_times)
me_values = me_values[valid]
me_times = me_times[valid]
```

iii. No additional processing beyond NaN removal and interpolation.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: discretized into 3 bins using GLOBAL quantile boundaries across all sessions.

ii.
```python
me_quantiles = np.array([-np.inf,
                          np.quantile(all_me, 1/3),
                          np.quantile(all_me, 2/3),
                          np.inf])
me_disc = np.digitize(sess['me_binned'], me_quantiles[1:-1]).astype(np.float32)
```

iii. Global discretization, same approach as wheel speed. The reference uses per-session percentiles.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same as wheel speed: interpolated using `interp1d` to the `np.linspace(t_start + binsize, t_end, n_bins)` grid, shifted by half a bin from the neural bin centers.

ii.
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
f_interp = interp1d(seg_times, seg_vals, kind='linear', fill_value='extrapolate')
result[trial_idx] = f_interp(x_interp)
```

iii. Same interpolation approach as wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with missing wheel data, whisker motion energy, or too few valid trials are skipped entirely. Within a session, trials where behavior interpolation fails (insufficient coverage) are dropped via `combined_valid`. NaN values in whisker ME are removed before interpolation. Sessions where no probe data loads are skipped. The code also loads additional trial timing files (`stimOnTrigger_times`, `goCueTrigger_times`) if available, and searches versioned subdirectories for files.

ii.
```python
combined_valid = wheel_valid & me_valid
if np.sum(combined_valid) < 2:
    return None
```

```python
valid = ~np.isnan(me_values) & ~np.isnan(me_times)
```

iii. The AI takes a conservative approach, dropping any session or trial with incomplete data.

## 10-a. What are the most time-consuming steps of the code?

i. Based on the timing output in the code, spike binning and wheel speed computation are the main per-session costs. Loading spike data from disk is also significant. The full conversion took approximately 35.5 minutes for 378 sessions.

ii.
```python
t1 = time.time()
binned_spikes = bin_spikes_fast(...)
t_spike = time.time() - t1
...
wheel_times, wheel_speed = compute_wheel_speed(sess_path)
...
print(f'  Session {eid}: ... (spike:{t_spike:.1f}s, wheel:{t_wheel:.1f}s, ME:{t_me:.1f}s, total:{t_total:.1f}s)')
```

iii. The code includes timing instrumentation for each major step.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates per-trial using `np.add.at` which could potentially be vectorized across trials. The behavior interpolation also loops per-trial. The `compute_trial_number_in_block` function uses an explicit Python loop over all trials.

ii.
```python
for trial_idx in range(n_trials):
    ...
    np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)
```

```python
for i in range(len(prob_left)):
    val = prob_left.iloc[i] if hasattr(prob_left, 'iloc') else prob_left[i]
```

iii. The per-trial loops are the main candidates for vectorization.

## 10-c. What processing does the code repeat multiple times?

i. The `BrainRegions()` object is instantiated inside `process_session` for every session rather than once globally. The wheel speed computation reimplements the interpolation and filtering that `SessionLoader.load_wheel()` already provides.

ii.
```python
br = BrainRegions()  # called for every session
```

iii. Creating a BrainRegions instance involves loading atlas data, which is repeated per session.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads additional trial timing files (`stimOnTrigger_times`, `goCueTrigger_times`) that are never used in the conversion. The `MAX_TRIAL_LEN` filter and NaN exclusion on `feedback_times` and `feedbackType` are unnecessary filters not present in the reference solution. The AI also loads `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` for ALL clusters including those that should have been filtered by quality.

ii.
```python
stim_trig_file = find_versioned_file(sess_path, '_ibl_trials.stimOnTrigger_times.npy')
if stim_trig_file is not None:
    trials_df['stimOnTrigger_times'] = np.load(stim_trig_file)

go_trig_file = find_versioned_file(sess_path, '_ibl_trials.goCueTrigger_times.npy')
```

```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times', 'probabilityLeft',
               'firstMovement_times', 'feedbackType']
```

iii. These extra data loads and filters add processing time without contributing to the final output.
