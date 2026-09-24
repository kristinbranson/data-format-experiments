# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `bwm_release.csv`, groups its probe rows by `eid`, resolves each session by lab/subject/date under `data/one_cache`, and directly loads parquet/NumPy files. Only locally resolvable sessions are attempted; sessions missing required streams are skipped.

ii.
```python
bwm_df = pd.read_csv(BWM_CSV, index_col=0)
for _, row in bwm_df.iterrows():
    sessions.setdefault(row.eid, []).append({...})
sess_path = find_session_path(probes[0]['lab'], probes[0]['subject'], probes[0]['date'])
```

iii. The notes say direct file loading matches the loader output and report 454 locally available sessions, with 378 surviving missing wheel/motion-energy data.

## 1-b. How are the data split into subjects?

i. Subject IDs come from the release CSV; successful sessions are indexed into a sorted unique subject list.

ii.
```python
all_subjects = sorted(set(sess['subject'] for sess in session_results))
subject_idx_list.append(subject_to_idx[sess['subject']])
```

iii. The release table provides the subject identity directly.

## 1-c. How are the data split into sessions?

i. Rows are grouped by the CSV `eid`; all probes sharing an `eid` form one session.

ii.
```python
for _, row in bwm_df.iterrows():
    if row.eid not in sessions: sessions[row.eid] = []
    sessions[row.eid].append({'probe_name': row.probe_name, ...})
```

iii. The agent treated `eid` as the canonical session identifier.

## 1-d. How are the data split into trials?

i. The trials parquet supplies one row per trial. Surviving rows define stimulus-aligned intervals and one output trial each.

ii.
```python
valid_trials = trials_df[trials_mask].copy()
stim_on = valid_trials[ALIGN_TIME].values
interval_starts = stim_on + TIME_WINDOW[0]
```

iii. The notes rely on the reference trial table and alignment logic.

## 1-e. How are trials filtered based on quality controls?

i. Trials require reaction time 0.08–2 s, duration at most 10 s where available, non-NaN listed fields, and nonzero choice. A second mask requires wheel and camera samples to cover the window within 20 ms. Sessions with fewer than two survivors are dropped.

ii.
```python
mask &= (rt >= MIN_RT) & (rt <= MAX_RT)
mask &= (trial_len <= MAX_TRIAL_LEN)
mask &= ~trials_df[event].isna()
mask &= (trials_df['choice'] != 0)
combined_valid = wheel_valid & me_valid
```

iii. The agent says these reproduce `load_trials_and_mask` defaults/reference calls and ensure complete behavior traces.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays derive from each probe's `spikes.times.npy` and `spikes.clusters.npy`; cluster/channel and channel atlas-ID arrays establish neuron count and regions.

ii.
```python
spike_times = np.load(spike_times_file).flatten()
spike_clusters = np.load(spike_clusters_file).flatten()
cluster_channels = np.load(spike_dir / 'clusters.channels.npy').flatten()
```

iii. The notes identify spike times and assignments as the reference neural source.

## 2-b. How is the `neural` data processed?

i. Probes are cluster-offset, concatenated, time-sorted, and spikes are counted into 100 20-ms bins per trial. Counts remain counts (not Hz).

ii.
```python
all_clusters.append(sc + cluster_offset)
idx_start = np.searchsorted(spike_times, t_start, side='left')
np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)
```

iii. The agent states this matches reference spike binning and validated one trial's raw and converted spike totals exactly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No cluster-quality or anatomical filter is applied; every cluster in the selected sorting is retained.

ii.
```python
n_clusters = len(cluster_channels)
cluster_brain_ids = chan_brain_ids[cluster_channels]
```

iii. The notes explicitly justify `qc=None` because `prepare_data` calls the reference loader without QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each window begins 0.5 s before `stimOn_times` and ends 1.5 s after it; spike bin indices are relative to that start.

ii.
```python
interval_starts = stim_on + TIME_WINDOW[0]
interval_ends = stim_on + TIME_WINDOW[1]
bin_idx = np.floor((trial_times - t_start) / binsize).astype(int)
```

iii. The notes cite the method configuration `align_time='stimOn_times'`, `(-0.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 20 ms, with 100 bins over two seconds. Spikes are binned once; no smoothing or later rebinning is performed.

ii.
```python
BINSIZE = 0.02
N_BINS = int(np.ceil(INTERVAL_LEN / BINSIZE))
```

iii. The papers/reference code use 20-ms bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is defined from `stimOn_times` plus the fixed window/bin grid rather than measured from another stream.

ii.
```python
stim_on = valid_trials[ALIGN_TIME].values
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

iii. The grid follows the reference behavior interpolation grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A 100-value linear grid from -0.48 through 1.5 s is generated and reused for every trial.

ii.
```python
np.linspace(-0.5 + 0.02, 1.5, 100).astype(np.float32)
```

iii. The agent calls these the reference bin centers.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Both use the same stimulus-relative 100-bin indexing, although the reported time points are right edges while spike bins span `[start,end)`.

ii.
```python
interval_starts = stim_on + TIME_WINDOW[0]
time_since_stim = np.linspace(TIME_WINDOW[0] + BINSIZE, TIME_WINDOW[1], N_BINS)
```

iii. The agent says the interpolation points match the reference code.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from consecutive equal values of `probabilityLeft` in the final filtered trial table.

ii.
```python
trial_num_in_block = compute_trial_number_in_block(valid_trials_final['probabilityLeft'])
```

iii. A block is understood as a run with constant left prior.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A Python loop counts from 1 and resets to 1 when prior changes. It runs after all trial filters, so removed trials do not advance the counter; the value is broadcast over 100 bins.

ii.
```python
count = 0
if val == current_block: count += 1
else: current_block, count = val, 1
np.full(N_BINS, sess['trial_num_in_block'][trial_idx])
```

iii. The code/docstring says trial numbering starts at 1; no rationale is given for recomputing after filtering.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It uses the trial-table `choice` column after excluding zero/no-choice trials.

ii.
```python
choice = valid_trials_final['choice'].values.copy()
```

iii. The notes identify this as the raw behavioral choice.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent maps raw -1 to output 0 (“left”) and everything else (+1) to 1 (“right”), then broadcasts it over time.

ii.
```python
choice_binary = np.where(choice == -1, 0, 1).astype(np.float32)
np.full(N_BINS, int(sess['choice_binary'][trial_idx]))
```

iii. The notes assert “left(-1)->0, right(1)->1,” which is the justification used, though it contradicts IBL's sign convention.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes directly from trial-table `probabilityLeft`.

ii.
```python
prob_left = valid_trials_final['probabilityLeft'].values.copy()
```

iii. This is the task's block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values 0.2, 0.5, and 0.8 map to 0, 1, and 2 and are broadcast across time.

ii.
```python
prior_cat[prob_left == 0.2] = 0
prior_cat[prob_left == 0.5] = 1
prior_cat[prob_left == 0.8] = 2
```

iii. This mapping is explicitly required by the instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It derives from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
position = np.load(pos_file).flatten()
timestamps = np.load(ts_file).flatten()
```

iii. The notes follow `SessionLoader.load_wheel` and take absolute velocity as speed.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is linearly interpolated to 1 kHz, low-pass filtered/differentiated with an order-8 20-Hz Butterworth procedure, absolutized, and linearly interpolated onto trial points.

ii.
```python
t_uniform = np.arange(timestamps[0], timestamps[-1], 1.0 / fs)
vel = np.insert(np.diff(scipy.signal.sosfiltfilt(sos, pos_interp)), 0, 0) * fs
speed = np.abs(vel)
```

iii. The agent says this reproduces IBL's standard wheel loader.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The 33rd and 67th percentiles are computed after pooling all retained values from all sessions; `digitize` yields low/medium/high.

ii.
```python
wheel_quantiles = np.array([-np.inf, np.quantile(all_wheel, 1/3),
                             np.quantile(all_wheel, 2/3), np.inf])
wheel_disc = np.digitize(sess['wheel_binned'], wheel_quantiles[1:-1])
```

iii. The notes justify equal-frequency global bins, but do not justify choosing global rather than per-session thresholds.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The shared session clock and stimulus windows are used; wheel values are interpolated at 100 points from window start +20 ms through the end.

ii.
```python
x_interp = np.linspace(t_start + binsize, t_end, n_bins)
result[trial_idx] = interp1d(seg_times, seg_vals, fill_value='extrapolate')(x_interp)
```

iii. The agent says this matches `get_behavior_per_interval`.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses left-camera ROI motion energy and camera times when present, otherwise right-camera equivalents.

ii.
```python
me_file = find_versioned_file(sess_path, 'leftCamera.ROIMotionEnergy.npy')
times_file = find_versioned_file(sess_path, '_ibl_leftCamera.times.npy')
```

iii. The notes say this follows the reference's left-first fallback.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Lengths are truncated to match, NaN samples removed, and the otherwise unfiltered trace is linearly interpolated to trial points.

ii.
```python
min_len = min(len(me_values), len(me_times))
valid = ~np.isnan(me_values) & ~np.isnan(me_times)
me_binned, me_valid = interpolate_behavior(...)
```

iii. The agent documents no extra motion-energy filtering.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Global pooled 33rd/67th percentiles define three categories.

ii.
```python
me_quantiles = np.array([-np.inf, np.quantile(all_me, 1/3),
                          np.quantile(all_me, 2/3), np.inf])
me_disc = np.digitize(sess['me_binned'], me_quantiles[1:-1])
```

iii. The notes describe global equal-frequency bins.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera data uses the same stimulus-relative windows and 100 interpolation points as wheel behavior.

ii.
```python
me_binned, me_valid = interpolate_behavior(
    me_times, me_values, interval_starts, interval_ends, BINSIZE, N_BINS)
```

iii. The agent says behavioral interpolation matches neural bins/reference processing.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing session paths/probes/behavior cause probes or sessions to be skipped; malformed sessions are caught; uncovered trials and NaNs are removed; streams of unequal motion-energy length are truncated; fewer than two usable trials drops a session.

ii.
```python
if result[0] is not None: probe_data.append(result)
if wheel_times is None: return None
combined_valid = wheel_valid & me_valid
except Exception as e: n_failed += 1
```

iii. Notes report 76 skipped sessions and characterize missing data as handled by exclusion.

## 10-a. What are the most time-consuming steps of the code?

i. Per-session spike loading/binning and wheel resampling/filtering dominate; the full serial conversion took about 35.5 minutes, and assembling/writing a 90-GB pickle is also substantial.

ii.
```python
binned_spikes = bin_spikes_fast(...)
wheel_times, wheel_speed = compute_wheel_speed(sess_path)
for i, (eid, probes) in enumerate(available_sessions.items()):
```

iii. The notes estimate about four seconds per session and record component timings.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial spike-binning and behavior-interpolation loops could be batched; block counting could use grouped cumulative counts. Final per-session/per-trial list construction could also be reduced, though the required nested format still needs object/list assembly.

ii.
```python
for trial_idx in range(n_trials):
    np.add.at(binned[trial_idx], (trial_clusters, bin_idx), 1)
for i in range(len(prob_left)):
    ...
```

iii. The agent labels spike binning “fast” using searchsorted/`add.at`, but gives no explicit vectorization analysis.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly searches filesystem/version directories, constructs a new `BrainRegions`, scans each behavior stream with `searchsorted` per trial, and loops over trials again to construct neural/input/output lists.

ii.
```python
for vd in versioned_dirs: ...
br = BrainRegions()
for trial_idx in range(n_trials): ...
```

iii. The agent did not explicitly justify these repetitions.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. An unused broken `merge_probes` loop and unused `bin_spikes_vectorized`/`discretize_to_bins` functions remain. Optional plots compute many summaries used only for diagnostics; behavior arrays are pooled into large global vectors solely to obtain two quantiles. Neural data are cast to float32 even though stored values are counts.

ii.
```python
for spike_times, spike_clusters, n_clusters, cluster_brain_ids in zip(*[iter(x) for x in [spikes_list]]):
    pass
all_wheel = np.concatenate(all_wheel)
```

iii. The notes present plots as validation and do not acknowledge the dead helpers or extra global pooling.
