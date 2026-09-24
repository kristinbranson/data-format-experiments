# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads `bwm_release.csv`, groups its probe rows by `eid`, constructs each cache path from lab/subject/date/session number, and directly opens NumPy/Parquet files. It does not use ONE or first restrict sessions to those with every required stream. It processes all 459 listed sessions, catching failures.

ii.
```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
session_groups = bwm_df.groupby('eid')
session_path = os.path.join(DATA_ROOT, lab, 'Subjects', subject, date, session_num_str)
```

iii. The notes say direct disk loading bypasses ONE because the cache is read-only, and the plan was to process all release sessions and skip missing/error cases.

## 1-b. How are the data split into subjects?

i. Subject identifiers come from the release CSV. During successful-session assembly, first-seen unique subject names are accumulated and each session receives that subject's index.

ii.
```python
row = group.iloc[0]
session_list.append((eid, row['lab'], row['subject'], row['date'],
                     row['session_number'], probe_names))
if subject not in subject_set:
    subject_set.append(subject)
subject_idx = subject_set.index(subject)
```

iii. The notes treat the CSV subject field as the dataset's subject identifier and validate the resulting subject count against the paper.

## 1-c. How are the data split into sessions?

i. Rows in the release CSV are grouped by `eid`; all probe names in a group belong to that session. The physical session directory is reconstructed from CSV metadata.

ii.
```python
for eid, group in bwm_df.groupby('eid'):
    probe_names = list(group['probe_name'])
```

iii. The notes identify 459 unique release sessions and regard the `eid` grouping as the session boundary.

## 1-d. How are the data split into trials?

i. The trials Parquet table supplies one row per trial. Each row's `stimOn_times` defines a two-second interval; binned neural and behavioral arrays are indexed by those rows and later subset by the common mask.

ii.
```python
trials_df = load_trials(session_path)
stim_times = trials_df[ALIGN_TIME].values
interval_begs = stim_times + TIME_WINDOW[0]
interval_ends = stim_times + TIME_WINDOW[1]
good_indices = np.where(combined_mask)[0]
```

iii. The AI followed the reference convention that the trials table is already trial-wise.

## 1-e. How are trials filtered based on quality controls?

i. Trials are removed for NaNs in six task fields, reaction time outside 0.08–2 s, feedback-minus-go-cue over 10 s, no choice, or failure of wheel/camera coverage checks. Sessions with fewer than two survivors are skipped.

ii.
```python
for event in NAN_EXCLUDE:
    mask &= ~trials_df[event].isna()
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
mask &= (trial_len <= MAX_TRIAL_LEN) | trial_len.isna()
mask &= (trials_df['choice'] != 0)
combined_mask = mask & wheel_mask & me_mask
```

iii. The notes say these reproduce `load_trials_and_mask()` and that behavior coverage is additionally necessary to form aligned outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from each probe's `spikes.times.npy` and `spikes.clusters.npy`; cluster channel/anatomy files are used only for region metadata.

ii.
```python
spikes = {'times': np.load(times_file).flatten(),
          'clusters': np.load(clusters_file).flatten()}
```

iii. The AI identified these as the reference spike-sorting inputs.

## 2-b. How is the `neural` data processed?

i. Probes are merged by offsetting cluster IDs and sorting spikes by time. Spikes are counted in 100 20-ms bins per trial using flattened indices and `np.bincount`. Counts are not divided by bin width and are ultimately cast to `uint8`.

ii.
```python
lin_idx = c * N_BINS + b
counts = np.bincount(lin_idx, minlength=minlength)
binned[trial_idx] = counts[:minlength].reshape(n_clusters_total, N_BINS)
neural_trial = neural_data[trial_idx].astype(np.uint8)
```

iii. The notes justify count binning as matching `bin_spiking_data()` and `uint8` as a memory optimization because most 20-ms counts are below 255.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered by cluster quality or anatomical validity: every cluster ID up to the maximum is retained, including possible gaps and `void` regions.

ii.
```python
n_clusters = int(merged_spikes['clusters'].max()) + 1
# no use of clusters['metrics'] to mask clusters
```

iii. The notes explicitly choose all clusters because the Zhang processing calls its loader with `qc=None`, despite acknowledging the data paper's well-isolated-unit curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial interval is `stimOn_times - 0.5` through `stimOn_times + 1.5`; absolute spike times are located with `searchsorted` and binned relative to the interval beginning.

ii.
```python
interval_begs = stim_times + TIME_WINDOW[0]
interval_ends = stim_times + TIME_WINDOW[1]
b = np.clip(((t - interval_begs[trial_idx]) / BINSIZE).astype(np.int32), 0, N_BINS - 1)
```

iii. The task explicitly requires stimulus-onset alignment, and the notes prefer the unified alignment in the caching code over analysis-specific paper windows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 20 ms, with 100 bins across two seconds. Spikes are directly histogrammed into those bins; there is no later neural rebinning or smoothing.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The AI cites the reference caching parameters and methods paper's 100-step representation.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is constructed from the configured window and bin size, conceptually relative to each trial's `stimOn_times`, rather than loaded as a raw column.

ii.
```python
time_since_onset = np.linspace(TIME_WINDOW[0] + BINSIZE,
                               TIME_WINDOW[1], N_BINS)
```

iii. The notes describe it as the interpolation/bin time axis dictated by stimulus alignment.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A 100-value linear grid from -0.48 s through +1.50 s is generated and copied into every trial.

ii.
```python
input_trial[0, :] = time_since_onset
```

iii. The AI says this matches the reference behavior interpolation targets (`interval_beg + binsize` through `interval_end`).

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The AI assigns one time value to each of the 100 neural bins, but labels the bins at their right edges (-0.48…1.50) while spike counts cover intervals starting at -0.50; thus it is consistently indexed but not at the reference bin centers.

ii.
```python
neural_trial = neural_data[trial_idx]
input_trial[0, :] = time_since_onset
```

iii. The AI believed the right-edge grid matched the reference helper.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the full trials table's `probabilityLeft` sequence; a value change marks a block boundary.

ii.
```python
all_prob_left = trials_df['probabilityLeft'].values
all_trial_nums = compute_trial_number_in_block(all_prob_left)
```

iii. The notes explain that no explicit block ID is needed because the prior is constant within blocks.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based counter resets whenever `probabilityLeft` changes. It is computed before filtering, subset afterward, and broadcast across all 100 time points.

ii.
```python
if prob_left[i] != current_val:
    current_block_start = i
trial_numbers[i] = i - current_block_start
input_trial[1, :] = trial_num_in_block[trial_idx]
```

iii. Computing it before filtering preserves the animal's actual position in the original block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived directly from `trials_df['choice']` after applying the trial mask.

ii.
```python
choice_raw = trials_df['choice'].values[good_indices]
```

iii. The AI correctly recognized the IBL choice column as the source, though its notes/code assign the signs differently from the human reference.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The formula maps raw -1 to class 0 and +1 to class 1, then broadcasts the class across the trial. This is opposite the human reference's +1→left/0 and -1→right/1 mapping.

ii.
```python
choice = ((choice_raw + 1) / 2).astype(np.int32)
output_trial[0, :] = choice[trial_idx]
```

iii. The AI states “-1 (left) → 0, 1 (right) → 1”; that sign convention is its justification.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from the trial table's `probabilityLeft` values after filtering.

ii.
```python
prob_left = trials_df['probabilityLeft'].values[good_indices]
```

iii. This is the task variable that directly represents the block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 are mapped to integer classes 0, 1, and 2 and broadcast across time.

ii.
```python
prior[prob_left == 0.2] = 0
prior[prob_left == 0.5] = 1
prior[prob_left == 0.8] = 2
```

iii. This mapping is explicitly required by the task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It comes from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
re_pos = np.load(pos_file).flatten()
re_ts = np.load(ts_file).flatten()
```

iii. The notes follow `SessionLoader.load_wheel()` and the reference's absolute wheel velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is linearly interpolated to 1 kHz, low-pass filtered with an order-8 20-Hz Butterworth filter, differentiated, multiplied by 1,000, and made absolute. Each trial is then linearly interpolated to 100 target times.

ii.
```python
position = scipy_interp1d(re_ts, re_pos, kind='linear')(t)
vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, position)), 0, 0) * fs
speed = np.abs(vel)
y_interp = interp1d(trial_times, trial_vals, kind='linear',
                    fill_value='extrapolate')(x_interp)
```

iii. The AI intentionally reproduces Brainbox's wheel processing rather than differentiating irregular raw samples.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. All retained wheel samples in a session are flattened; the 33⅓ and 66⅔ percentiles define three equal-frequency classes via `np.digitize`.

ii.
```python
thresholds = np.percentile(valid, np.linspace(0, 100, n_bins + 1))
result = np.digitize(values, thresholds[1:-1], right=False)
```

iii. The notes choose session-wide quantiles to produce balanced categorical decoder targets.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. It uses the same stimulus-relative trial interval and 100 indices, but interpolation targets run from interval start +20 ms through the interval end (right edges), not the reference neural bin centers.

ii.
```python
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
binned_wheel, wheel_mask = interpolate_behavior(...)
```

iii. The AI says this matches `get_behavior_per_interval()` and uses coverage checks at both edges.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `leftCamera.ROIMotionEnergy.npy` with `_ibl_leftCamera.times.npy`, falling back to the corresponding right-camera files.

ii.
```python
me_file = find_file(alf_path, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_file(alf_path, '_ibl_leftCamera.times.npy')
```

iii. The left-then-right preference follows the reference loading logic.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy trace is not filtered or normalized. It is linearly interpolated to the 100 per-trial targets and then discretized session-wide.

ii.
```python
binned_me, me_mask = interpolate_behavior(me_times, me_values,
                                           interval_begs, interval_ends)
me_discrete = discretize_to_bins(me_data.flatten(), n_bins=3)
```

iii. The notes state that no additional motion-energy processing is required.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The retained session trace is flattened and split at its 33⅓ and 66⅔ percentiles into labels 0, 1, and 2.

ii.
```python
me_discrete_2d = discretize_to_bins(me_flat, n_bins=3).reshape(me_data.shape)
```

iii. Equal-frequency bins were chosen to balance the three decoder classes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera timestamps are interpolated on each stimulus-relative 100-point grid. As with wheel, those targets are right edges (-0.48…1.50), rather than the human reference's bin centers.

ii.
```python
idxs_beg = np.searchsorted(beh_times, interval_begs, side='right')
x_interp = np.linspace(t_beg + BINSIZE, t_end, N_BINS)
```

iii. The AI relied on synchronized session clocks and the reference helper's coverage checks.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. File loaders return `None` for missing/invalid files; probes without spikes are skipped, sessions with no spikes or fewer than two valid trials are dropped, and all session exceptions are logged and skipped. Behavioral gaps invalidate trials. The newest lexicographically sorted revision is selected. Missing anatomy becomes `void`.

ii.
```python
if spikes is not None:
    spikes_list.append(spikes)
if n_good_trials < 2:
    return None
except Exception as e:
    print(f"  ERROR processing {eid}: {type(e).__name__}: {e}")
    continue
```

iii. The notes say 21 sessions had zero valid trials and were skipped; robust completion of the full release was preferred to aborting on missing data.

## 10-a. What are the most time-consuming steps of the code?

i. Loading large spike arrays and binning spikes dominate; full conversion also incurs large temporary/final pickle I/O. The notes estimate roughly 0.9 s/session loading spikes and 0.4 s/session binning in the sample, with a 36-minute full run.

ii.
```python
spikes, clusters = load_spike_data(session_path, pname)
binned_spikes = bin_spikes_fast(...)
pickle.dump(..., f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The timings and the memory-failure investigation in the notes identify these costs empirically.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial spike binning and behavior interpolation could be vectorized further; block numbering could use pandas grouping; per-neuron region indexing and per-trial formatting could use mappings/array operations. The inner per-spike loop exists only in an unused slow alternative; the executed spike routine already vectorizes spikes within each trial.

ii.
```python
for i, trial_idx in enumerate(valid_idx):
    ...
for trial_idx in range(n_trials):
    ...
for neuron_idx in range(result['n_neurons']):
    ...
```

iii. The notes emphasize that `searchsorted` boundaries and `np.bincount` were vectorized for a measured speedup, but do not discuss the remaining loops in detail.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly searches/copies trial windows for spikes, wheel, and motion energy; scans list-backed subject/region sets with membership plus `.index`; serializes each session to a temporary pickle and later reloads it (including an initial metadata pass and a subsequent data pass); and broadcasts constant per-trial values into 100 columns.

ii.
```python
if region not in region_set:
    region_set.append(region)
session_region_idx[neuron_idx] = region_set.index(region)
with open(tmp_file, 'rb') as f:
    sess = pickle.load(f)
```

iii. Temporary serialization/reloading was deliberately added after in-memory accumulation exceeded the 64-GB limit; repetition was accepted to control peak memory.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads cluster depths and metrics but never uses them; computes/collects some metadata arrays not needed for neural values; defines an unused slow `bin_spikes_vectorized`; and optional plotting computes large summaries used only for diagnostics. It also bins all trials before applying the quality/behavior mask, so rejected trials' spike counts are discarded.

ii.
```python
clusters['depths'] = np.load(depths_file).flatten()
clusters['metrics'] = pd.read_parquet(metrics_file)
binned_spikes = bin_spikes_fast(...)
neural_data = binned_spikes[good_indices]
```

iii. Metrics/depths were explored for QC/anatomy but the explicit all-cluster decision made metrics unnecessary; all-trial binning simplified alignment and masking.
