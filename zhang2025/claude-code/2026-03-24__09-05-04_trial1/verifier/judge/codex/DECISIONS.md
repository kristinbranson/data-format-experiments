# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the session list from `bwm_release.csv`, then tries to find each session under the local ONE cache and reads raw files directly from `alf/` rather than using the ONE API. It searches revision folders and picks the latest matching file. A key implementation detail is that session lookup is hard-coded to `001`, so sessions whose `session_number` is not 1 are missed.

ii.
```python
DATA_DIR = Path('/app/data/one_cache')
BWM_CSV = Path('/app/code/code_zhang2025/data/bwm_release.csv')

def find_session_path(lab, subject, date, number=1):
    sess_path = DATA_DIR / lab / 'Subjects' / subject / date / f'{number:03d}' / 'alf'
    if sess_path.exists():
        return sess_path
    return None
```

```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
sessions = get_session_list(bwm_df)

for idx in range(len(sessions)):
    sess = sessions.iloc[idx]
    result = process_session(sess, br, show_processing=args.show_processing)
```

iii. In `CONVERSION_NOTES.md` Step 4 and Step 5, the AI says it used the BWM release CSV as the ground-truth session list and chose direct file loading from the cache instead of the ONE API because the data were already local. The trajectory also records that it wanted an offline implementation that still matched the reference data sources.

## 1-b. How are the data split into subjects?

i. Subjects are split by the `subject` field attached to each successfully processed session. After session processing, the AI takes the sorted unique subject names and builds `subject_idx` so each session points to one subject entry.

ii.
```python
all_subjects = sorted(set(r['subject'] for r in all_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}

for r in all_results:
    subject_idx.append(subject_to_idx[r['subject']])
```

```python
data = {
    'subjects': all_subjects,
    'subject_idx': np.array(subject_idx),
    ...
}
```

iii. `CONVERSION_NOTES.md` Step 2 identifies `subject` as part of the BWM session metadata, and Step 9 reports final subject counts from processed sessions. The AI’s notes consistently treat subject grouping as a postprocessing step over session metadata rather than a separate loading stage.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values from `bwm_release.csv`. The AI groups the CSV by `eid`, keeps one row per session, and processes each session independently into one entry of `neural`, `input`, and `output`.

ii.
```python
def get_session_list(bwm_df):
    sessions = bwm_df.groupby('eid').first().reset_index()
    return sessions
```

```python
for idx in range(len(sessions)):
    sess = sessions.iloc[idx]
    result = process_session(sess, br, show_processing=args.show_processing)
```

iii. In `CONVERSION_NOTES.md` Step 4, the AI explicitly says it resolved the session-count discrepancy by treating the BWM release CSV as the ground truth, matching the reference code path in `0_data_caching.py`.

## 1-d. How are the data split into trials?

i. Within each session, the trials are loaded from `_ibl_trials.table.pqt`, filtered with a trial mask, and then further filtered by whether wheel and whisker signals are available for the aligned interval. The remaining rows define the per-session trial list.

ii.
```python
def load_trials(alf_path):
    trials_file = find_latest_revision(alf_path, '_ibl_trials.table.pqt')
    trials = pd.read_parquet(trials_file)
    return trials
```

```python
trials = load_trials(alf_path)
mask = create_trial_mask(trials)
valid_trials = trials[mask].reset_index(drop=True)
...
final_trials = valid_trials[combined_mask].reset_index(drop=True)
```

iii. The AI’s notes say this mirrors the reference pattern of loading a full trials table and then applying masks. The justification is that the trial table is the authoritative source for session segmentation and trial-alignment events.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering happens in two stages. First, the AI reproduces the reference trial mask: drop trials with NaNs in key columns, reaction times outside `[0.08, 2.0]`, no-choice trials, and trials longer than 10 s. Second, it drops trials whose wheel or whisker interpolations contain NaNs anywhere in the aligned window.

ii.
```python
NAN_EXCLUDE = ['stimOn_times', 'choice', 'feedback_times',
               'probabilityLeft', 'firstMovement_times', 'feedbackType']

def create_trial_mask(trials):
    mask = pd.Series(True, index=trials.index)
    for event in NAN_EXCLUDE:
        if event in trials.columns:
            mask &= ~trials[event].isna()
    rt = trials['firstMovement_times'] - trials['stimOn_times']
    mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
    mask &= (trials['choice'] != 0)
    if 'goCue_times' in trials.columns:
        trial_len = trials['feedback_times'] - trials['goCue_times']
        mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()
    return mask
```

```python
combined_mask = np.ones(len(valid_trials), dtype=bool)
if wheel_speed_binned is not None:
    combined_mask &= ~np.any(np.isnan(wheel_speed_binned), axis=1)
else:
    combined_mask[:] = False

if me_binned is not None:
    combined_mask &= ~np.any(np.isnan(me_binned), axis=1)
else:
    combined_mask[:] = False
```

iii. `CONVERSION_NOTES.md` Step 1 and Step 3 explicitly list the reference trial-mask criteria, and Step 9 notes that sessions with missing wheel data or too few fully valid trials were skipped entirely.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from spike times and cluster assignments from each probe, plus cluster-to-channel and channel-to-brain-region information for metadata. The core arrays are `spikes.times.npy` and `spikes.clusters.npy`.

ii.
```python
spikes = {
    'times': np.load(rev_dir / 'spikes.times.npy').flatten(),
    'clusters': np.load(rev_dir / 'spikes.clusters.npy').flatten(),
}

clusters_channels = np.load(rev_dir / 'clusters.channels.npy').flatten()
clusters_depths = np.load(rev_dir / 'clusters.depths.npy').flatten()
chan_brain_ids = np.load(rev_dir / 'channels.brainLocationIds_ccf_2017.npy').flatten()
```

iii. The AI’s notes identify these as the same pykilosort outputs used by the Zhang reference code via `SpikeSortingLoader`, just loaded directly from cache files.

## 2-b. How is the `neural` data processed?

i. The AI merges probes by offsetting cluster and channel indices, sorts merged spikes by time, maps clusters to Beryl brain regions, bins spikes into 20 ms trial-aligned bins, and stores each trial as an `(n_neurons, n_timepoints)` spike-count matrix. It then casts the per-trial matrices to `uint8` to save memory.

ii.
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
n_clusters = len(clusters['channels'])
cluster_regions = get_brain_regions(clusters, br)

binned_spikes = bin_spikes_vectorized(
    spikes['times'], spikes['clusters'], n_clusters,
    interval_starts, interval_ends, BINSIZE, N_BINS
)
```

```python
neural_trials = []
for t in range(len(final_spikes)):
    neural_trials.append(final_spikes[t].astype(np.uint8))
```

iii. `CONVERSION_NOTES.md` Step 5 says the AI intended to match `bin_spiking_data()` / `get_spike_data_per_interval()` from the reference code. The trajectory later records the explicit `uint8` decision as a memory optimization after confirming the decoder re-casts neural data to `float32` during training.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered by cluster quality at all. The AI uses every cluster returned by spike sorting, equivalent to `qc=None` in the reference implementation.

ii.
```python
def load_spike_sorting(alf_path, probe_name):
    ...
    spikes = {
        'times': np.load(rev_dir / 'spikes.times.npy').flatten(),
        'clusters': np.load(rev_dir / 'spikes.clusters.npy').flatten(),
    }
    ...
    return spikes, clusters
```

iii. In `CONVERSION_NOTES.md` Step 4, the AI explicitly resolves the neuron-filtering discrepancy by following the reference code rather than the data paper’s “well-isolated only” count.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned to `stimOn_times`, with a window from `-0.5 s` to `+1.5 s` around stimulus onset.

ii.
```python
ALIGN_TIME = 'stimOn_times'
TIME_WINDOW = (-0.5, 1.5)

align_times = valid_trials[ALIGN_TIME].values
interval_starts = align_times + TIME_WINDOW[0]
interval_ends = align_times + TIME_WINDOW[1]
```

iii. The AI’s notes say this follows the task instruction to align to stimulus onset and also matches the `0_data_caching.py` parameterization in the reference repo.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins throughout the full aligned window, giving 100 bins per trial. No further temporal rebinning is applied.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. `CONVERSION_NOTES.md` Step 3 and Step 4 both say the AI chose the reference code’s 20 ms bins rather than the 50 ms choice/prior bins mentioned elsewhere in the paper.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. This input is constructed from the alignment window and bin size, not read from a dedicated raw variable. Conceptually it is derived from `stimOn_times` plus the fixed relative bin grid.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. In Step 5 of the notes, the AI describes this as a manually constructed time-varying input because the decoder task asked for “time since stimulus onset” rather than a trial-table column.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI builds a 100-point linear grid from `-0.48` to `1.5` seconds, corresponding to the aligned 20 ms bins. It uses the same right-edge/bin-center convention as the behavioral interpolation grid.

ii.
```python
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. The AI justifies this in the notes by pointing to the reference interpolation convention `np.linspace(interval_beg + binsize, interval_end, n_bins)` used for behavior streams.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same 100-bin time vector is used for every trial, and it is inserted as the first row of the per-trial `input` array so it is exactly aligned to the neural binning.

ii.
```python
inp = np.array([
    time_since_stim,
    np.full(N_BINS, trial_in_block_final[t], dtype=np.float32),
], dtype=np.float32)
```

iii. The AI’s notes say all decoder inputs and outputs should share the same `stimOn_times`-aligned time axis for compatibility with downstream decoding.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column of the trials table by detecting block transitions whenever `probabilityLeft` changes.

ii.
```python
prob_left_all = trials['probabilityLeft'].values
trial_in_block_all = compute_trial_in_block(prob_left_all)
```

```python
def compute_trial_in_block(prob_left):
    trial_in_block = np.zeros(len(prob_left), dtype=np.float32)
    count = 1
    for i in range(len(prob_left)):
        if i > 0 and prob_left[i] != prob_left[i-1]:
            count = 1
        trial_in_block[i] = count
        count += 1
    return trial_in_block
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly states that the AI computes trial number in block from changes in `probabilityLeft` and resets the count at each block boundary.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI scans all trials in order, resets the counter when the block probability changes, then applies the trial-quality and behavior-availability masks to keep only the surviving trials. The final scalar is broadcast across time bins within each trial.

ii.
```python
trial_in_block_valid = trial_in_block_all[mask.values]
trial_in_block_final = trial_in_block_valid[combined_mask]
```

```python
np.full(N_BINS, trial_in_block_final[t], dtype=np.float32)
```

iii. The AI’s notes say this choice preserved the original trial order within blocks even after trial exclusion, which it considered the intended block-context signal for the decoder.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived directly from the `choice` column of the trials table.

ii.
```python
choice = final_trials['choice'].values.copy()
```

iii. The AI’s notes list `trials.choice` as the source field and note that no-choice trials are removed earlier by the trial mask.

## 5-b. What processing is involved in computing `output` *Choice*?

i. IBL encodes left as `-1` and right as `1`. The AI remaps this to `0` for left and `1` for right, then broadcasts the per-trial category across all 100 time bins.

ii.
```python
choice_encoded = ((choice + 1) // 2).astype(int)
```

```python
np.full(N_BINS, choice_encoded[t], dtype=int)
```

iii. `CONVERSION_NOTES.md` Step 4 says the task specification overrode the original sign convention and required `left = 0`, `right = 1`.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the per-trial `probabilityLeft` column in the trials table.

ii.
```python
prob_left = final_trials['probabilityLeft'].values
```

iii. The AI’s notes say it treated block probability as the required categorical prior because the task specification explicitly named the `0.2`, `0.5`, and `0.8` levels.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps the raw block probabilities to class labels `0`, `1`, and `2`, corresponding to `0.2`, `0.5`, and `0.8`, then broadcasts the per-trial value across all time bins.

ii.
```python
prior_encoded = np.zeros(len(prob_left), dtype=int)
prior_encoded[prob_left == 0.2] = 0
prior_encoded[prob_left == 0.5] = 1
prior_encoded[prob_left == 0.8] = 2
```

```python
np.full(N_BINS, prior_encoded[t], dtype=int)
```

iii. `CONVERSION_NOTES.md` Step 4 explicitly records that the AI chose the categorical task-spec encoding instead of reconstructing the continuous prior estimate discussed in the methods paper.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
wheel_pos_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.position.npy')).flatten()
wheel_ts_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.timestamps.npy')).flatten()
wheel_times, wheel_speed = interpolate_wheel(wheel_ts_raw, wheel_pos_raw)
```

iii. The AI’s notes say this matches the same underlying source used by `SessionLoader.load_wheel()` in the reference utilities.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. Wheel position is linearly interpolated to 1 kHz, filtered with an 8th-order Butterworth low-pass filter with a 20 Hz corner, differentiated to obtain velocity, converted to speed via absolute value, and then re-interpolated into the 100 stimulus-aligned trial bins.

ii.
```python
def interpolate_wheel(timestamps, position, fs=WHEEL_FS, corner_freq=WHEEL_CORNER_FREQ, order=WHEEL_FILTER_ORDER):
    t = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
    pos_interp = interp1d(timestamps, position, kind='linear')(t)
    sos = signal.butter(N=order, Wn=corner_freq / fs * 2, btype='lowpass', output='sos')
    vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
    return t, np.abs(vel).astype(np.float32)
```

```python
wheel_speed_binned, wheel_good = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

iii. In Step 6 of the notes, the AI says it intentionally matched ibllib’s wheel preprocessing. That is also reflected in the trajectory summary, which explicitly mentions “1kHz interp, 20Hz corner, order 8”.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI flattens all wheel-speed time points from a session, computes two quantile cut points, and assigns every time point to low, medium, or high speed based on those session-specific tertiles.

ii.
```python
def discretize_to_bins(values, n_bins=N_DISCRETE_BINS):
    valid = values[~np.isnan(values)]
    quantiles = np.linspace(0, 100, n_bins + 1)
    boundaries = np.percentile(valid, quantiles)
    result = np.digitize(values, boundaries[1:-1])
    result = np.clip(result, 0, n_bins - 1)
    return result.astype(int), boundaries
```

```python
all_wheel_flat = final_wheel.flatten()
wheel_disc, wheel_boundaries = discretize_to_bins(all_wheel_flat, N_DISCRETE_BINS)
wheel_disc_2d = wheel_disc.reshape(final_wheel.shape).astype(int)
```

iii. `CONVERSION_NOTES.md` Step 5 justifies this as an equal-frequency discretization so the three classes are balanced within a session, and Step 10 reports the expected `33/33/33%` split.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. After computing continuous speed on the wheel timeline, the AI interpolates it to the same `stimOn_times`-aligned `[-0.5, 1.5]` grid used for spikes. Trials with incomplete wheel coverage are discarded.

ii.
```python
wheel_speed_binned, wheel_good = interpolate_behavior_to_bins(
    wheel_times, wheel_speed, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

```python
combined_mask &= ~np.any(np.isnan(wheel_speed_binned), axis=1)
```

iii. The notes say this choice followed the task’s request to align the dataset to stimulus onset even though the methods paper used movement-aligned windows for some behavior decoders.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` plus `_ibl_leftCamera.times.npy`, with fallback to the right camera equivalents if the left camera files are missing.

ii.
```python
if side == 'left':
    me_file = find_latest_revision(alf_path, 'leftCamera.ROIMotionEnergy.npy')
    times_file = find_latest_revision(alf_path, '_ibl_leftCamera.times.npy')
else:
    me_file = find_latest_revision(alf_path, 'rightCamera.ROIMotionEnergy.npy')
    times_file = find_latest_revision(alf_path, '_ibl_rightCamera.times.npy')
```

```python
me_times, me_values = load_motion_energy(alf_path, side='left')
if me_times is None:
    me_times, me_values = load_motion_energy(alf_path, side='right')
```

iii. `CONVERSION_NOTES.md` Step 6 and the trajectory summary both say the AI intentionally used left-camera whisker motion energy with right-camera fallback, matching the reference behavior loader.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI loads the raw motion-energy trace, truncates time and value arrays to the shorter length if they differ, and linearly interpolates the trace into the common 100-bin trial-aligned window.

ii.
```python
me = np.load(me_file).flatten()
times = np.load(times_file).flatten()
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len].astype(np.float32)
```

```python
me_binned, me_good = interpolate_behavior_to_bins(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

iii. The AI’s notes explicitly mention handling a common length mismatch in camera streams by truncation. That was its documented strategy for small raw-data inconsistencies.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same per-session tertile discretization as wheel speed: flatten all time points, compute quantile boundaries, and digitize into `low`, `medium`, and `high`.

ii.
```python
all_me_flat = final_me.flatten()
me_disc, me_boundaries = discretize_to_bins(all_me_flat, N_DISCRETE_BINS)
me_disc_2d = me_disc.reshape(final_me.shape).astype(int)
```

iii. `CONVERSION_NOTES.md` Step 5 says the AI used the same equal-frequency discretization for wheel speed and whisker motion energy to satisfy the 3-class decoder output requirement.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The AI interpolates whisker motion energy onto the same `stimOn_times`-aligned 100-bin grid as the spikes and wheel speed, and it drops trials with incomplete whisker coverage.

ii.
```python
me_binned, me_good = interpolate_behavior_to_bins(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS
)
```

```python
combined_mask &= ~np.any(np.isnan(me_binned), axis=1)
```

iii. As with wheel speed, the AI’s justification in the notes is that the final decoder dataset needed one common stimulus-aligned time axis across all variables.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or inconsistent data by a mix of truncation, fallback, masking, and session skipping:
- missing revisioned files: search latest revision first
- mismatched motion-energy array lengths: truncate to the shorter length
- missing left whisker data: fall back to right whisker data
- NaNs or out-of-range reaction times in trials: exclude those trials
- missing wheel or whisker coverage for aligned bins: drop those trials, and skip the session if fewer than two trials remain
- missing session paths or unreadable key files: skip the session

ii.
```python
def find_latest_revision(base_path, filename_pattern):
    ...
```

```python
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len].astype(np.float32)
```

```python
if me_times is None:
    me_times, me_values = load_motion_energy(alf_path, side='right')
...
if n_valid < 2:
    print(f"  Session {eid}: fewer than 2 trials with all data, skipping")
    return None
```

iii. `CONVERSION_NOTES.md` Step 9 and Step 10 summarize this policy and explicitly note that 67 sessions were skipped for missing data paths, missing wheel data, or too few valid trials.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive steps are the per-session processing loop, spike binning across all trials and neurons, wheel interpolation/filtering, and behavior interpolation for wheel and whisker motion energy.

ii.
```python
for idx in range(len(sessions)):
    sess = sessions.iloc[idx]
    result = process_session(sess, br, show_processing=args.show_processing)
```

```python
for trial_idx in range(n_trials):
    ...
    np.add.at(binned[trial_idx].ravel(), flat_idx, 1)
```

```python
for trial_idx in range(n_trials):
    ...
    y_interp = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. The AI’s notes report about 20 s per session in sample mode and 29 minutes for the full run, and the trajectory shows it explicitly profiled memory/time to identify spike storage and session processing as bottlenecks.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining vectorization opportunities are:
- the per-trial loop in `bin_spikes_vectorized`
- the per-trial loop in `interpolate_behavior_to_bins`
- the Python loop in `compute_trial_in_block`
- the Python loops that build `neural_trials`, `input_trials`, and `output_trials`

ii.
```python
for trial_idx in range(n_trials):
    ...
```

```python
for i in range(len(prob_left)):
    if i > 0 and prob_left[i] != prob_left[i-1]:
        count = 1
```

```python
for t in range(len(final_trials)):
    neural_trials.append(final_spikes[t].astype(np.uint8))
...
for t in range(len(final_trials)):
    input_trials.append(inp)
...
for t in range(len(final_trials)):
    output_trials.append(out)
```

iii. The AI repeatedly says in the notes and trajectory that it optimized some heavy paths already, but the final code still keeps several trial-wise Python loops for simplicity.

## 12-c. What processing does the code repeat multiple times?

i. The code repeats several similar operations:
- directory scanning through `find_latest_revision` for many files in every session
- nearly identical trial-wise interpolation logic for wheel and whisker streams
- repeated per-trial broadcasting with `np.full` for scalar inputs/outputs
- flatten/discretize/reshape performed separately for wheel speed and whisker motion energy

ii.
```python
wheel_speed_binned, wheel_good = interpolate_behavior_to_bins(...)
...
me_binned, me_good = interpolate_behavior_to_bins(...)
```

```python
np.full(N_BINS, trial_in_block_final[t], dtype=np.float32)
np.full(N_BINS, choice_encoded[t], dtype=int)
np.full(N_BINS, prior_encoded[t], dtype=int)
```

iii. The AI did not single out these repetitions in one note section, but they follow directly from the final `convert_data.py` structure and are consistent with the implementation choices documented in Step 6.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computed or loaded items are not used downstream:
- `clusters.metrics.pqt` is loaded but not used in conversion decisions
- `wheel_good` and `me_good` are computed but never read
- `wheel_boundaries` and `me_boundaries` are computed but never used or saved
- `ProcessPoolExecutor` / `as_completed` are imported but unused
- optional processing plots are generated only for inspection and not used by the decoder

ii.
```python
metrics_file = rev_dir / 'clusters.metrics.pqt'
if metrics_file.exists():
    metrics = pd.read_parquet(metrics_file)
else:
    metrics = None
```

```python
wheel_disc, wheel_boundaries = discretize_to_bins(all_wheel_flat, N_DISCRETE_BINS)
...
me_disc, me_boundaries = discretize_to_bins(all_me_flat, N_DISCRETE_BINS)
```

```python
wheel_speed_binned, wheel_good = interpolate_behavior_to_bins(...)
...
me_binned, me_good = interpolate_behavior_to_bins(...)
```

iii. The AI’s trajectory shows it added some of this work for validation and debugging during development, especially around plotting and memory checking, even though the final decoder dataset only needs the final discrete arrays and metadata.
