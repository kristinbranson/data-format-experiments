# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `bwm_release.csv`, groups its probe rows by `eid`, constructs each cache path from lab/subject/date, and directly loads Parquet/NumPy files with pandas, NumPy, and glob. It iterates all 459 release sessions, skipping missing or unusable ones; the run retained 340.

ii.
```python
bwm_df = pd.read_csv('code/code_zhang2025/data/bwm_release.csv', index_col=0)
session_groups = bwm_df.groupby('eid').agg({'lab': 'first', 'subject': 'first',
    'date': 'first', 'probe_name': list, 'pid': list}).reset_index()
session_dir = find_session_dir(lab, subject, date)
```

iii. The notes say the cache is organized by lab/Subjects/subject/date/001 and report 459 release sessions but only 341 with every required stream. The agent chose direct local loading rather than ONE because it had explored the cache layout.

## 1-b. How are the data split into subjects?

i. The `subject` column from the release CSV identifies mice. A first-seen-order map builds `subjects`, and each retained session gets the corresponding `subject_idx`.

ii.
```python
subj = result['subject']
if subj not in subject_map:
    subject_map[subj] = len(all_subjects)
    all_subjects.append(subj)
subject_idx_list.append(subject_map[subj])
```

iii. The notes treat the release metadata's subject name as the unique mouse identifier; no identifier is inferred from signal files.

## 1-c. How are the data split into sessions?

i. Rows of `bwm_release.csv` are grouped by `eid`; probe names become a list, so one processed item is one behavioral session with all its probes merged.

ii.
```python
session_groups = bwm_df.groupby('eid').agg({
    'lab': 'first', 'subject': 'first', 'date': 'first',
    'probe_name': list, 'pid': list}).reset_index()
for sess_idx, (_, row) in enumerate(session_groups.iterrows()):
```

iii. The notes identify `eid` as session-level metadata and explicitly note that sessions can contain one or two probes, which should be merged.

## 1-d. How are the data split into trials?

i. Each trials-table row is a trial. Its stimulus onset defines `[-0.5, 1.5)` seconds; spikes and behavioral streams are sliced/interpolated into 100 bins for every row before the final mask is applied.

ii.
```python
stim_on = trials_df[ALIGN_TIME].values
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]
binned_spikes = bin_spikes_per_trial(spike_times, spike_clusters, n_clusters,
                                     trial_starts, trial_ends)
```

iii. The agent says this matches the reference's two-second, stimulus-aligned trial construction.

## 1-e. How are trials filtered based on quality controls?

i. Trials require reaction time 0.08–2 s, feedback-minus-go-cue at most 10 s, nonmissing key fields, nonzero choice, and at least two usable samples in both behavior windows. Sessions with fewer than two surviving trials are dropped.

ii.
```python
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
mask &= (trial_len <= MAX_TRIAL_LEN)
for event in nan_exclude:
    mask &= ~trials_df[event].isna()
mask &= (trials_df['choice'] != 0)
combined_mask = mask.values & wheel_valid & me_valid
```

iii. The notes attribute these cuts to `load_trials_and_mask` and explain that both time-varying outputs are required, so trials without usable wheel or camera data are removed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from each probe's `spikes.times.npy` and `spikes.clusters.npy`; `clusters.channels.npy` and channel CCF IDs are used only to attach Beryl region labels.

ii.
```python
spike_times = np.load(os.path.join(ks_dir, 'spikes.times.npy')).flatten()
spike_clusters = np.load(os.path.join(ks_dir, 'spikes.clusters.npy')).flatten()
clusters_channels = np.load(os.path.join(ks_dir, 'clusters.channels.npy')).flatten()
```

iii. The agent identified the same spike arrays and Beryl mapping in the reference code.

## 2-b. How is the `neural` data processed?

i. Probe spike trains are offset into one session-wide cluster index, concatenated and time-sorted. Spikes are counted per cluster in 20-ms trial bins. The script stores counts (float32 in source; the produced pickle was later mechanically converted to uint8), not firing rates.

ii.
```python
spike_clusters = spike_clusters + cluster_offset
merged_times = np.concatenate(all_spike_times)
sort_idx = np.argsort(merged_times, kind='stable')
np.add.at(binned_all[trial_idx], (clust_curr, time_bin_idx), 1)
```

iii. The notes state that the reference bins spikes without smoothing and uses all clusters. The trajectory later justifies uint8 because the observed maximum count was 68, reducing the pickle size fourfold.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not quality-filtered: every cluster in `clusters.channels.npy` is retained, including `root`/`void` regions; missing anatomy becomes `unknown`.

ii.
```python
n_clusters = len(clusters_channels)
beryl = np.array(['unknown'] * n_clusters)  # when anatomy is absent
all_spike_times.append(spike_times)
all_spike_clusters.append(spike_clusters)
```

iii. The notes explicitly chose all clusters because the explored methods code calls spike loading with `qc=None`, distinguishing that from the paper's well-isolated-neuron statistic.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute stimulus onset is shifted by −0.5 and +1.5 s to form each window. Spike bins are indexed relative to the window start, putting onset at the boundary between bins 24 and 25.

ii.
```python
trial_starts = stim_on + TIME_WINDOW[0]
trial_ends = stim_on + TIME_WINDOW[1]
time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)
```

iii. The notes resolve differing paper alignments in favor of the explicit task instruction and caching code: align every stream to `stimOn_times`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 20 ms: 100 bins over two seconds. Spikes are directly counted in those bins; there is no later neural rebinning or smoothing.

ii.
```python
BINSIZE = 0.02
TIME_WINDOW = (-0.5, 1.5)
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The agent cites the methods paper and reference parameters (`binsize=0.02`, 100 time steps).

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is defined from `stimOn_times` through the fixed relative trial grid, rather than measured from a separate raw stream.

ii.
```python
stim_on = trials_df[ALIGN_TIME].values
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
```

iii. The agent chose the requested stimulus event and the reference's −0.5-to-1.5-s window.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The 100 bin centers are generated from −0.49 through 1.49 s, cast to float32, and repeated identically for every trial.

ii.
```python
time_since_stim = np.arange(N_BINS) * BINSIZE + TIME_WINDOW[0] + BINSIZE / 2
time_since_stim = time_since_stim.astype(np.float32)
```

iii. No additional justification is given beyond matching the binning parameters.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Its entries are the centers of the exact bins used for the spike-count columns, so input column `t` corresponds to neural column `t`.

ii.
```python
time_bin_idx = np.floor((times_curr - t_beg) / BINSIZE).astype(np.int32)
inp = np.vstack([time_since_stim[np.newaxis, :],
                 np.full((1, N_BINS), trial_num_in_block[i])])
```

iii. The shared stimulus-aligned window and 20-ms grid were intended to align every modality.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the full unfiltered sequence of `probabilityLeft`; a change marks a new block.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
trial_num_in_block = all_trial_nums[valid_idx]
```

iii. The agent notes it must be computed before filtering so removed trials still count toward the animal's actual block position.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A loop resets a counter to 1 when the prior changes and increments it thereafter; thus indexing is one-based. The scalar is broadcast across 100 bins.

ii.
```python
counter = 1
for i in range(len(prob_left)):
    if i > 0 and prob_left[i] != prob_left[i-1]: counter = 1
    trial_nums[i] = counter
    counter += 1
```

iii. The docstring explicitly says the trial number resets to 1. No rationale is given for differing from a zero-based `cumcount`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from the trials table's `choice` column; zero-choice trials have already been removed.

ii.
```python
choice = valid_trials['choice'].values.copy()
choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
```

iii. The agent intended a binary left=0/right=1 mapping, but did not document IBL's sign convention in the notes.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code maps raw −1 to class 0 and every surviving other value (+1) to class 1, then broadcasts the class over all 100 time bins.

ii.
```python
choice_mapped = np.where(choice == -1, 0, 1).astype(np.int64)
np.full((1, N_BINS), choice_mapped[i], dtype=np.int64)
```

iii. Comments assert “−1 -> 0 (left), 1 -> 1 (right).” This is the agent's stated rationale, although it reverses the IBL convention used by the human reference.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes directly from the trials table's `probabilityLeft` column.

ii.
```python
prob_left = valid_trials['probabilityLeft'].values
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
```

iii. The notes identify `probabilityLeft` as the block prior and follow the mapping stated in the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are mapped to 0, 1, and 2; any unexpected value silently defaults to 1. The result is constant over the trial's time bins.

ii.
```python
prior_mapped = np.array([prior_map.get(p, 1) for p in prob_left], dtype=np.int64)
np.full((1, N_BINS), prior_mapped[i], dtype=np.int64)
```

iii. The explicit three-class mapping is required by the task. The notes do not justify the fallback, though earlier NaN exclusion makes normal inputs one of the expected values.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
wheel_pos = np.load(pos_file).flatten()
wheel_ts = np.load(ts_file).flatten()
```

iii. The notes say the reference defines wheel speed as the absolute value of velocity computed from those arrays.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is linearly interpolated to 1 kHz, low-pass filtered with an order-8 20-Hz Butterworth filter, differentiated and absolutized. The trace is then linearly interpolated/extrapolated to trial-bin centers.

ii.
```python
pos_interp = interpolate.interp1d(wheel_ts, wheel_pos, kind='linear')(t_interp)
sos = signal.butter(N=8, Wn=20 / 1000 * 2, btype='lowpass', output='sos')
vel = np.insert(np.diff(signal.sosfiltfilt(sos, pos_interp)), 0, 0) * 1000
speed = np.abs(vel)
```

iii. The agent attributes this pipeline to IBL's `wheel.py` and `SessionLoader` defaults.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Across all retained wheel samples in a session, the 1/3 and 2/3 quantiles are computed and `np.digitize` assigns classes 0–2.

ii.
```python
quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
boundaries = np.quantile(flat, quantiles)
result = np.digitize(values, boundaries).astype(np.int64)
```

iii. The notes call for three quantile-based bins, producing approximately equal-sized within-session classes.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is evaluated at the same 100 absolute bin-center times from stimulus onset as the spike bins.

ii.
```python
x_interp = np.linspace(t_beg, t_end, N_BINS, endpoint=False) + BINSIZE / 2
y_interp = interpolate.interp1d(beh_t_clean, beh_v_clean,
    kind='linear', fill_value='extrapolate')(x_interp)
```

iii. The agent chose common stimulus alignment for all variables as required by the task.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses left-camera ROI motion energy and left-camera timestamps when valid, otherwise the corresponding right-camera pair.

ii.
```python
me_file = find_file(session_dir, 'leftCamera.ROIMotionEnergy.npy')
cam_file = find_file(session_dir, '_ibl_leftCamera.times.npy')
# then analogous right-camera files
```

iii. The notes say this matches the reference's left-first, right-fallback behavior.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy trace is flattened without filtering or normalization, cleaned of NaNs within each window, and linearly interpolated/extrapolated to the neural bin centers.

ii.
```python
me = np.load(me_file).flatten()
nan_mask = ~np.isnan(beh_v)
y_interp = interpolate.interp1d(beh_t_clean, beh_v_clean,
    kind='linear', fill_value='extrapolate')(x_interp)
```

iii. The notes specify no transformation beyond bin-time interpolation and later discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It independently uses the session's 1/3 and 2/3 quantiles across all retained trial/time samples, then digitizes into 0, 1, 2.

ii.
```python
me_disc = discretize_time_varying(me_data, N_DISC_BINS)
boundaries = np.quantile(flat, [1/3, 2/3])
result = np.digitize(values, boundaries).astype(np.int64)
```

iii. The three equal-frequency categories implement the required discretization and mirror the wheel rule.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera values are interpolated at the same 100 stimulus-relative bin centers used by the neural matrix.

ii.
```python
me_binned, me_valid = interpolate_behavior_to_bins(
    me_times, me_values, trial_starts, trial_ends)
me_data = me_binned[valid_idx]
```

iii. The agent states that task-level stimulus alignment overrides the behavior-specific alignment mentioned elsewhere in the paper.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing session directories, trials, spikes, wheel, or motion energy cause sessions or all their trials to be skipped. Length mismatches and too-short streams are rejected. Within a behavior window, NaNs are removed if two samples remain; interpolation may extrapolate to edge bin centers. Exceptions during a session are caught and the session is skipped.

ii.
```python
if len(wheel_pos) != len(wheel_ts) or len(wheel_pos) < 10: return None, None
if nan_mask.sum() < 2: valid[trial_idx] = False
try: result = process_session(...)
except Exception as e:
    n_skipped += 1
    continue
```

iii. The trajectory explains the 340-session result as the complete intersection of required streams and accepts dropping incomplete data because all four requested outputs must be present.

## 10-a. What are the most time-consuming steps of the code?

i. Full conversion is dominated by loading/sorting large spike arrays and per-trial spike binning; wheel interpolation/filtering and pickle serialization are also substantial. Sessions are processed serially. Later decoder SVD/training was far slower but is outside conversion itself.

ii.
```python
merged_times = np.concatenate(all_spike_times)
sort_idx = np.argsort(merged_times, kind='stable')
for trial_idx in valid_indices:
    np.add.at(binned_all[trial_idx], (clust_curr, time_bin_idx), 1)
```

iii. Runtime logging was added around spike binning and entire sessions; the trajectory repeatedly identifies the huge all-cluster dataset as the cause of long conversion, serialization, and training times.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial spike binning and behavior interpolation, the trial-number counter, per-trial input/output construction, session iteration, prior mapping, and neuron-to-region mapping could be vectorized or parallelized. The most important are the serial session loop and repeated per-trial work.

ii.
```python
for trial_idx in valid_indices: ...
for trial_idx in range(n_trials): ...
for i in range(n_valid_trials): input_list.append(...)
for i in range(n_valid_trials): output_list.append(...)
for sess_idx, (_, row) in enumerate(session_groups.iterrows()): ...
```

iii. The function calls spike binning “optimized vectorized,” citing `searchsorted` and `np.add.at`, but it still loops over trials. No justification is given for serial session processing.

## 10-c. What processing does the code repeat multiple times?

i. Each trial window is searched twice for behavior (wheel and whisker), and all trials are traversed separately for spike binning, both behaviors, input construction, and output construction. File discovery repeatedly globs similar cache directories, and region values are mapped neuron by neuron during assembly.

ii.
```python
wheel_binned, wheel_valid = interpolate_behavior_to_bins(...)
me_binned, me_valid = interpolate_behavior_to_bins(...)
for i in range(n_valid_trials): ...  # inputs
for i in range(n_valid_trials): ...  # outputs
```

iii. The agent does not explicitly discuss repeated work; its documentation focuses on correctness and observed runtime.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It bins spikes and interpolates both behaviors for every raw trial before applying the quality mask, so work for rejected trials is discarded. It also reads/aggregates unused `pid`, computes unused local variables, keeps session EIDs only as metadata, imports `bincount2D` without using it, and initially materialized/saved float32 neural counts before later converting the pickle to uint8.

ii.
```python
# Bin spikes per trial (for ALL trials first, then filter)
binned_spikes = bin_spikes_per_trial(...)
combined_mask = mask.values & wheel_valid & me_valid
neural_data = binned_spikes[valid_idx]
```

iii. The agent explicitly says it processed all trials first and later optimized storage after observing an 84.7-GB file; it gives no scientific reason for doing rejected-trial work.
