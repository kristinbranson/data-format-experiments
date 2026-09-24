# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the Brain-Wide Map release CSV, reduces it to one row per `eid`, constructs a cache path from lab/subject/date (always session `001`), and directly loads parquet/NumPy files. It processes the 459 listed sessions sequentially; sessions with missing paths or required streams are skipped, leaving 392.

ii.
```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
sessions = get_session_list(bwm_df)
...
sess_path = DATA_DIR / lab / 'Subjects' / subject / date / f'{number:03d}' / 'alf'
```

iii. The notes say the BWM CSV is the ground truth because it matches the paper's 459 sessions, and direct file loading avoids requiring the ONE API offline. They accept the 67 skipped sessions as missing data.

## 1-b. How are the data split into subjects?

i. Subject names come from the release CSV. After conversion, unique names are sorted and each retained session receives an index into that list.

ii.
```python
all_subjects = sorted(set(r['subject'] for r in all_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
subject_idx.append(subject_to_idx[r['subject']])
```

iii. The release metadata already provides subject identifiers, so the agent does not infer them from neural files.

## 1-c. How are the data split into sessions?

i. The release CSV is grouped by `eid`; the first metadata row represents each session. Each is then processed independently.

ii.
```python
def get_session_list(bwm_df):
    sessions = bwm_df.groupby('eid').first().reset_index()
    return sessions
```

iii. The notes identify the CSV's 459 unique sessions as the paper-consistent session list.

## 1-d. How are the data split into trials?

i. Each row of `_ibl_trials.table.pqt` is treated as a trial. Valid rows provide stimulus-onset-aligned start and end times, and neural/behavior streams are sliced or interpolated for each interval.

ii.
```python
trials = pd.read_parquet(trials_file)
align_times = valid_trials[ALIGN_TIME].values
interval_starts = align_times + TIME_WINDOW[0]
interval_ends = align_times + TIME_WINDOW[1]
```

iii. This follows the reference trial table and its `stimOn_times` alignment field.

## 1-e. How are trials filtered based on quality controls?

i. Trials with NaNs in six key columns, reaction times outside 0.08–2 s, choice 0, or trial lengths over 10 s are excluded. Later, trials without complete wheel and camera coverage are also removed, and sessions with fewer than two survivors are skipped.

ii.
```python
for event in NAN_EXCLUDE:
    mask &= ~trials[event].isna()
rt = trials['firstMovement_times'] - trials['stimOn_times']
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
mask &= (trials['choice'] != 0)
trial_len = trials['feedback_times'] - trials['goCue_times']
mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()
...
combined_mask &= ~np.any(np.isnan(wheel_speed_binned), axis=1)
combined_mask &= ~np.any(np.isnan(me_binned), axis=1)
```

iii. The agent says this mask matches the supplied repository code: NaN exclusion, RT limits, no-choice removal, a 10 s maximum, and neural/behavior trial alignment.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `spikes.times.npy` and `spikes.clusters.npy` on every available probe. Cluster channel and atlas arrays provide neuron count and region labels.

ii.
```python
spikes = {
    'times': np.load(rev_dir / 'spikes.times.npy').flatten(),
    'clusters': np.load(rev_dir / 'spikes.clusters.npy').flatten(),
}
clusters_channels = np.load(rev_dir / 'clusters.channels.npy').flatten()
```

iii. The notes map these variables to the reference `load_spiking_data` and binning functions.

## 2-b. How is the `neural` data processed?

i. Probes are merged by offsetting cluster/channel indices and sorting spikes by time. Spikes are counted per cluster in 100 20-ms bins for every trial. Counts are stored as `uint8`; they are not divided by bin width into Hz.

ii.
```python
flat_idx = c_sel[valid] * n_bins + bin_idx[valid]
np.add.at(binned[trial_idx].ravel(), flat_idx, 1)
...
neural_trials.append(final_spikes[t].astype(np.uint8))
```

iii. The agent says reference code bins spikes without smoothing and chose `uint8` because 20-ms spike counts are small, reducing the full pickle enough to fit memory.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not quality-filtered: all sorted clusters are retained, including clusters whose quality label is below 1 and atlas `void` units.

ii.
```python
n_clusters = len(clusters['channels'])
# no metrics-label or region mask is applied before binning
binned_spikes = bin_spikes_vectorized(
    spikes['times'], spikes['clusters'], n_clusters, ...)
```

iii. The notes explicitly resolve the paper/code discrepancy in favor of the methods repository's `qc=None`, despite documenting that the data paper's stringent `label >= 1` cut yields 75,708 well-isolated neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial window runs from 0.5 s before to 1.5 s after `stimOn_times`. Absolute spike times are selected in that interval and binned relative to its start.

ii.
```python
align_times = valid_trials[ALIGN_TIME].values
interval_starts = align_times + TIME_WINDOW[0]
interval_ends = align_times + TIME_WINDOW[1]
bin_idx = ((t_sel - t_start) / binsize).astype(np.int32)
```

iii. The task requests stimulus-onset alignment, which the agent treats as overriding variable-specific alignments in the paper.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Neural activity is newly binned at 20 ms into 100 bins over a 2 s window. No smoothing or later temporal rebinning is applied.

ii.
```python
BINSIZE = 0.02
TIME_WINDOW = (-0.5, 1.5)
N_BINS = int(np.ceil((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE))
```

iii. The notes cite the reference configuration of 20-ms bins and 100 time steps.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is constructed from the fixed window and bin size around each trial's `stimOn_times`; no measured continuous raw signal is used.

ii.
```python
ALIGN_TIME = 'stimOn_times'
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE,
                              TIME_WINDOW[1], N_BINS).astype(np.float32)
```

iii. The agent says it manually constructs the shared time axis from the same reference decoding parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. One linearly spaced vector from -0.48 through 1.50 s is created and reused for all trials. These are bin right edges, not bin centers.

ii.
```python
time_since_stim = np.linspace(-0.5 + 0.02, 1.5, 100).astype(np.float32)
```

iii. The notes describe the intended values inconsistently as `-0.5` to `1.48`; the implementation and behavior interpolation actually use right edges.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It has 100 samples corresponding in order to the 100 neural count bins, but labels each bin by its right edge rather than its center.

ii.
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

iii. The agent intended neural, inputs, wheel, and motion energy to share exactly one grid, and validation confirmed matching dimensions.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the complete, unfiltered `probabilityLeft` sequence; a change in probability marks a block boundary.

ii.
```python
prob_left_all = trials['probabilityLeft'].values
trial_in_block_all = compute_trial_in_block(prob_left_all)
```

iii. The task table has no explicit block ID, so the agent reconstructs blocks from the blockwise-constant prior before filtering trials.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A Python loop resets the counter to 1 when the prior changes and otherwise increments it. Thus numbering is one-based, and later-excluded trials still advance the counter. The scalar is broadcast over 100 time bins.

ii.
```python
count = 1
for i in range(len(prob_left)):
    if i > 0 and prob_left[i] != prob_left[i-1]:
        count = 1
    trial_in_block[i] = count
    count += 1
```

iii. The notes justify preserving real position by computing before filtering, but do not justify the departure from the reference's zero-based `cumcount()`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from `trials.choice`; no-choice rows are removed first.

ii.
```python
choice = final_trials['choice'].values.copy()
```

iii. The agent relies on the IBL trial-table choice code, but its stated sign convention is reversed relative to the reference solution.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The formula maps raw -1 to category 0 labeled left and raw +1 to category 1 labeled right, then broadcasts that class across the trial.

ii.
```python
# IBL: -1=left, 1=right -> convert to 0=left, 1=right
choice_encoded = ((choice + 1) // 2).astype(int)
```

iii. The agent says this satisfies the requested left=0/right=1 mapping. The human reference identifies the actual IBL convention as +1 left and -1 right, so the labels are inverted.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes directly from `trials.probabilityLeft` for retained trials.

ii.
```python
prob_left = final_trials['probabilityLeft'].values
```

iii. The notes follow the task's explicit use of the block prior rather than a running estimate.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are encoded as 0, 1, and 2 and broadcast across all bins.

ii.
```python
prior_encoded[prob_left == 0.2] = 0
prior_encoded[prob_left == 0.5] = 1
prior_encoded[prob_left == 0.8] = 2
```

iii. This mapping is explicitly required by the task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
wheel_pos_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.position.npy')).flatten()
wheel_ts_raw = np.load(find_latest_revision(alf_path, '_ibl_wheel.timestamps.npy')).flatten()
```

iii. The agent follows the repository's use of wheel position/timestamps and absolute velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is linearly interpolated to 1 kHz, low-pass filtered with an order-8 20-Hz Butterworth filter, differentiated, multiplied by sampling rate, and converted to absolute velocity. It is then linearly interpolated to trial bins.

ii.
```python
pos_interp = interp1d(timestamps, position, kind='linear')(t)
sos = signal.butter(N=order, Wn=corner_freq / fs * 2, btype='lowpass', output='sos')
vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
return t, np.abs(vel).astype(np.float32)
```

iii. The notes say these parameters reproduce ibllib/`SessionLoader` wheel processing.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. After incomplete trials are removed, all wheel samples in a session are flattened. The 33.3rd and 66.7th percentiles define low/medium/high classes.

ii.
```python
all_wheel_flat = final_wheel.flatten()
wheel_disc, wheel_boundaries = discretize_to_bins(all_wheel_flat, 3)
boundaries = np.percentile(valid, np.linspace(0, 100, n_bins + 1))
result = np.digitize(values, boundaries[1:-1])
```

iii. Equal-frequency session-level bins satisfy the categorical-output requirement and match the human approach.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is sampled at 100 points from trial start +20 ms through trial end, on the same ordered bins as neural activity, using linear interpolation/extrapolation.

ii.
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
y_interp = interp1d(t_sel, v_sel, kind='linear', fill_value='extrapolate')(x_interp)
```

iii. The agent intended the task's stimulus-onset alignment for all streams, though it uses right edges rather than the reference's bin centers.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `leftCamera.ROIMotionEnergy.npy` plus `_ibl_leftCamera.times.npy`, falling back to the corresponding right-camera files.

ii.
```python
me_times, me_values = load_motion_energy(alf_path, side='left')
if me_times is None:
    me_times, me_values = load_motion_energy(alf_path, side='right')
```

iii. This follows the available side-camera ROI motion-energy data and the left-preferred reference behavior.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released one-dimensional motion-energy trace is truncated with its time vector to their shorter length, cast to float32, and linearly interpolated to the trial grid without filtering or normalization.

ii.
```python
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len].astype(np.float32)
```

iii. The notes say the released trace requires no additional processing; truncation handles a common IBL length mismatch.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. All retained samples in a session are flattened and thresholded at session 1/3 and 2/3 quantiles into three classes.

ii.
```python
all_me_flat = final_me.flatten()
me_disc, me_boundaries = discretize_to_bins(all_me_flat, N_DISCRETE_BINS)
```

iii. The notes choose equal-frequency session-level low/medium/high categories, the same approach as wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera motion energy is interpolated at the same 100 stimulus-aligned right-edge timestamps used for wheel and the time input; incomplete windows are removed.

ii.
```python
me_binned, me_good = interpolate_behavior_to_bins(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS)
```

iii. Shared session clocks and common interpolation times are the agent's alignment rationale.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing session paths, trials, probes, wheel, or camera data cause probes/sessions to be skipped. Motion-energy/time length mismatch is silently truncated. Trials with NaNs or incomplete behavior coverage are dropped. Exceptions are logged and conversion continues.

ii.
```python
except Exception as e:
    print(f"  Session {eid}, {probe_name}: error loading spikes: {e}")
    continue
...
min_len = min(len(me), len(times))
return times[:min_len], me[:min_len].astype(np.float32)
```

iii. The full-run notes attribute 67 skipped sessions to missing files/data and regard 392/459 sessions as acceptable; truncation is described as handling a common IBL mismatch.

## 10-a. What are the most time-consuming steps of the code?

i. Per-session spike-file loading/merging and per-trial spike binning dominate, followed by 1-kHz wheel interpolation/filtering. Sessions are processed serially despite importing process-pool utilities; the run took 1,741 s.

ii.
```python
for idx in range(len(sessions)):
    result = process_session(sess, br, show_processing=args.show_processing)
...
for trial_idx in range(n_trials):
    idx_beg = np.searchsorted(spike_times, t_start, side='left')
```

iii. Sample timing was estimated at about 20 s/session and 2.5 hours total; the final run was about 29 minutes. The notes do not provide a component-level profile.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial loops in spike binning and behavior interpolation, the loop computing trial position, and per-trial array construction could be vectorized. More importantly, independent sessions could be parallelized with the already imported `ProcessPoolExecutor`.

ii.
```python
for trial_idx in range(n_trials):
    ...
for i in range(len(prob_left)):
    ...
for t in range(len(final_trials)):
    input_trials.append(inp)
```

iii. The agent calls spike binning “vectorized” because accumulation within each trial uses flat NumPy indices, but retains the outer Python loops and does not discuss them in the notes.

## 10-c. What processing does the code repeat multiple times?

i. `find_latest_revision` repeatedly scans the same ALF/revision directories for trials, each spike/cluster field, wheel arrays, and camera arrays. Trial-wise creation also repeatedly allocates identical time vectors and broadcast constants.

ii.
```python
find_latest_revision(probe_path, 'spikes.times.npy')
find_latest_revision(alf_path, '_ibl_wheel.position.npy')
find_latest_revision(alf_path, '_ibl_wheel.timestamps.npy')
...
np.full(N_BINS, choice_encoded[t], dtype=int)
```

iii. No explicit justification or caching strategy is documented.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads cluster depths and metrics but never uses them for filtering or output. It bins spikes for behavior-incomplete trials and only afterward discards those neural trials. Quantile boundary arrays are computed but not saved. Optional plotting calculates summaries that do not affect conversion.

ii.
```python
clusters_depths = np.load(rev_dir / 'clusters.depths.npy').flatten()
metrics = pd.read_parquet(metrics_file)
...
binned_spikes = bin_spikes_vectorized(...)
final_spikes = binned_spikes[combined_mask]
...
wheel_disc, wheel_boundaries = discretize_to_bins(...)
```

iii. Metrics/depth loading appears inherited from anticipated QC/region processing; the notes do not justify the discarded work.
