# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent takes the 459-session roster and probe list from `bwm_release.csv`, constructs each cached session path, and reads local parquet/NumPy ALF files directly. Full mode submits every roster row to session workers; sessions that cannot produce at least two jointly valid trials are omitted.

ii.
```python
bwm_df = pd.read_csv(BWM_RELEASE_CSV, index_col=0)
sessions_df = bwm_df[["eid", "lab", "subject", "date", "session_number"]].drop_duplicates("eid")
...
records = sessions_df.to_dict(orient="records")
executor.submit(process_one_session_worker, row_dict, time_axis)
```

iii. The notes call this roster the canonical release used by the reference code. Direct local reads were chosen to avoid slow/transient ONE calls and unnecessary AP metadata while retaining the same source variables.

## 1-b. How are the data split into subjects?

i. The `subject` column of the release CSV identifies each mouse. Assembly preserves first-seen unique subject order and creates one subject index per retained session.

ii.
```python
subjects = ordered_unique([session.subject for session in processed_sessions])
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
subject_idx.append(subject_to_idx[session.subject])
```

iii. The notes state that release metadata already provides the subject and deterministic session ordering.

## 1-c. How are the data split into sessions?

i. Each distinct `eid` in `bwm_release.csv` is a session; probe rows are grouped by `eid`, and each session is processed independently before results are restored to release order.

ii.
```python
.drop_duplicates(subset=["eid"], keep="first")
...
bwm_df[["eid", "pid", "probe_name"]].groupby("eid").apply(...)
processed_sessions.sort(key=lambda session: session.release_index)
```

iii. The agent regarded the 459-session methods-code roster as canonical and parallelized independent sessions for speed.

## 1-d. How are the data split into trials?

i. The trials parquet already has one row per trial. Neural and continuous behavior are segmented into `stimOn_times + [-0.5, 1.5]`, and retained row indices select matching trial arrays.

ii.
```python
intervals = np.vstack([trials_df[PARAMS["align_time"]] - .5,
                       trials_df[PARAMS["align_time"]] + 1.5]).T
keep_idx = np.flatnonzero(keep_mask)
neural_trials = [binned_spikes[i].T for i in keep_idx]
```

iii. The notes say this reproduces the executable reference code's common stimulus-aligned two-second trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are rejected for reaction time below 80 ms or above 2 s, `feedback-goCue > 10 s`, missing required events, no choice, or incomplete wheel/whisker coverage. A session must retain at least two trials.

ii.
```python
query_parts += ["(firstMovement_times - stimOn_times < 0.08)",
                "(firstMovement_times - stimOn_times > 2.0)",
                "(feedback_times - goCue_times > 10.0)"]
...
keep_mask = np.asarray(trials_mask) & wheel_mask & whisker_mask
if keep_idx.size < 2: raise RuntimeError(...)
```

iii. The agent explicitly chose the reference-code mask, including its 10 s rule, and added joint behavior coverage because all requested outputs must exist.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from per-probe `spikes.times.npy` and `spikes.clusters.npy`; cluster metrics, cluster channels, and channel atlas IDs supply QC and anatomical metadata.

ii.
```python
spikes = {"times": np.load(sort_dir / "spikes.times.npy"),
          "clusters": np.load(sort_dir / "spikes.clusters.npy")}
clusters_labeled = pd.read_parquet(sort_dir / "clusters.metrics.pqt")
cluster_channels = np.load(sort_dir / "clusters.channels.npy")
```

iii. The notes identify this as electrophysiology, so no calcium or delta-F/F processing is needed.

## 2-b. How is the `neural` data processed?

i. QC-passing clusters from all probes are merged and renumbered, then spikes are histogrammed into 100 nonoverlapping 20 ms bins for every trial. The stored matrices are neuron-by-time `float32` spike counts; they are not divided by bin width.

ii.
```python
spikes, clusters = merge_probes(spikes_list, clusters_list)
binned_tmp, _, cluster_idxs = bincount2D(..., xbin=binsize, xlim=[t_beg, t_end])
neural_trials = [binned_spikes[i].T.astype(np.float32, copy=False) for i in keep_idx]
```

iii. The agent says it reused the reference-style spike-binning logic and describes the result throughout its notes and metadata as spike counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` are loaded. Unlike the human solution, no explicit Beryl `void` exclusion is made.

ii.
```python
iok = clusters_labeled["label"] >= qc
...
load_spiking_data_current(..., qc=1)
```

iii. The agent chose well-isolated units to match the data paper's approximately 75,708-unit scale, although it noted the methods caching helper itself did not filter before binning.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each neural interval begins 0.5 s before and ends 1.5 s after `stimOn_times`; absolute spike timestamps are binned within those intervals.

ii.
```python
intervals = np.vstack([trials_df["stimOn_times"] - 0.5,
                       trials_df["stimOn_times"] + 1.5]).T
```

iii. The agent prioritized the explicit stimulus-onset instruction and the executable reference cache configuration.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 100 bins of 20 ms across two seconds. Raw spikes are binned once; there is no further rebinning or smoothing.

ii.
```python
PARAMS = {"binsize": 0.02, "time_window": (-0.5, 1.5)}
n_bins = int(np.ceil(interval_len / binsize))
```

iii. The notes cite the reference code's 20 ms common grid.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the configured `stimOn_times` alignment window and bin size, rather than a separate raw signal.

ii.
```python
start, end = PARAMS["time_window"]
binsize = PARAMS["binsize"]
return np.linspace(start + binsize, end, n_bins, dtype=np.float32)
```

iii. The agent says the required input is not in the raw table and must be derived directly from the adopted grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. It creates 100 bin-end values from -0.48 through 1.50 s and repeats that vector for every retained trial.

ii.
```python
time_axis = np.linspace(start + binsize, end, n_bins, dtype=np.float32)
np.vstack([time_axis, np.full(time_axis.shape[0], block_num_all[i])])
```

iii. A code comment claims bin ends match the reference behavior interpolation grid; the notes report the resulting range `[-0.48, 1.50]`.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It has one value per neural bin, but labels each `[t,t+0.02)` neural bin by its right edge rather than its center.

ii.
```python
x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
time_axis = build_time_axis()
```

iii. The agent considered the shared number of bins and stimulus window sufficient alignment and visually checked for off-by-one effects.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from changes in the complete, unfiltered `trials.probabilityLeft` sequence.

ii.
```python
prob_left_all = trials_df["probabilityLeft"].to_numpy(dtype=float)
block_num_all = trial_number_in_block(prob_left_all)
```

iii. The agent notes that the trial table has no explicit block counter, so the constant block prior defines boundaries.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A loop resets the counter to 1 whenever the prior changes and increments it thereafter. It is computed before filtering, then the retained values are broadcast over time.

ii.
```python
if idx == 0 or not np.isclose(value, prev): current = 1
else: current += 1
out[idx] = current
```

iii. Computing before filtering preserves true experimental progression. The agent did not justify choosing one-based rather than the human solution's zero-based count.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from `trials.choice`.

ii.
```python
choice_all = trials_df["choice"].to_numpy()
choice_labels = choice_to_label(choice_all[keep_idx])
```

iii. The notes cite official IBL semantics for the raw sign.

## 5-b. What processing is involved in computing `output` *Choice*?

i. No-choice trials are filtered; raw `+1` maps to left/0 and `-1` to right/1, then the per-trial label is repeated over 100 bins.

ii.
```python
return (choice_values == -1).astype(np.int64)
np.full(T, session.choice_labels[trial_idx], dtype=np.int64)
```

iii. This implements the requested binary left/right coding exactly.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials.probabilityLeft`.

ii.
```python
prior_labels = prior_to_label(prob_left_all[keep_idx])
```

iii. The agent identifies this column as the task's block prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values are rounded to one decimal, mapped `0.2/0.5/0.8 -> 0/1/2`, validated, and repeated over time.

ii.
```python
rounded = np.round(np.asarray(probability_left, dtype=float), 1)
return np.array([PRIOR_MAP[x] for x in rounded], dtype=np.int64)
```

iii. The mapping is explicitly required by the task; rounding guards against floating-point representation.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
wheel_timestamps = np.load(..."_ibl_wheel.timestamps.npy")
wheel_position = np.load(..."_ibl_wheel.position.npy")
```

iii. The agent selected the same wheel variables used by the reference behavior loader.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is interpolated at 1 kHz, converted to velocity with an eighth-order 20 Hz low-pass filter, absolutized, then linearly interpolated at 100 trial-relative bin-end times.

ii.
```python
position, times = interpolate_position(wheel_timestamps, wheel_position, freq=1000)
velocity, _ = velocity_filtered(position, fs=1000, corner_frequency=20, order=8)
...
fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
```

iii. The notes say this preserves reference loading/alignment before task-required categorization.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Two thresholds are computed at the 1/3 and 2/3 quantiles of every retained wheel sample pooled across all converted sessions; `np.digitize` maps values to 0, 1, or 2.

ii.
```python
wheel_values = np.concatenate([np.concatenate(s.wheel_continuous) for s in processed_sessions])
q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
return np.digitize(values, bins=np.array([q1, q2]), right=False)
```

iii. The agent deliberately chose global tertiles for consistent thresholds and exactly balanced aggregate classes.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel is evaluated on the same stimulus-relative two-second window and has 100 samples, but those samples are at bin ends (-0.48 to 1.50), whereas neural values are counts over bins beginning at -0.50.

ii.
```python
x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
```

iii. The agent considered this the reference interpolation convention and reported visual checks with no apparent shift.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `<side>Camera.ROIMotionEnergy.npy` and matching `_ibl_<side>Camera.times.npy`, preferring left and falling back to right.

ii.
```python
target = load_target_behavior_current(session_path, "left-whisker-motion-energy")
if target.get("skip"):
    target = load_target_behavior_current(session_path, "right-whisker-motion-energy")
```

iii. The notes say this camera fallback matches the provided code.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Released motion energy is used without filtering. Excess leading camera timestamps are trimmed, and the trace is linearly interpolated onto the 100 bin-end times.

ii.
```python
if times.shape[0] > values.shape[0]: times = times[-values.shape[0]:]
fn = interp1d(t_seg, v_seg, kind="linear", fill_value="extrapolate")
y_interp = fn(x_interp)
```

iii. The agent says no extra transformation is needed before alignment and categorization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. As for wheel, thresholds are global 1/3 and 2/3 quantiles pooled over every retained session and time point.

ii.
```python
whisker_values = np.concatenate([np.concatenate(s.whisker_continuous) for s in processed_sessions])
q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
```

iii. Global tertiles were chosen to apply one rule consistently and produce balanced aggregate categories.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It uses the same `stimOn_times + [-0.5,1.5]` trial window and 100 values, evaluated at bin ends rather than neural-bin centers.

ii.
```python
align_times = trials_df[PARAMS["align_time"]].to_numpy()
x_interp = np.linspace(interval_begs[idx] + binsize, interval_ends[idx], n_bins)
```

iii. The agent considered the common clock/window aligned and relied on visual checks.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/invalid events are trial-filtered; incomplete behavior windows are dropped; left-camera failure falls back to right; unequal camera lengths are trimmed only when timestamps are longer; transient session failures are retried three times and then sequentially; sessions with fewer than two valid trials or no good clusters are skipped.

ii.
```python
if times.shape[0] > values.shape[0]: times = times[-values.shape[0]:]
keep_mask = trials_mask & wheel_mask & whisker_mask
for attempt in range(1, 4): ...
if keep_idx.size < 2: raise RuntimeError(...)
```

iii. The notes emphasize robust completion without inventing missing values and document 21 sessions excluded because no jointly valid trials survived.

## 10-a. What are the most time-consuming steps of the code?

i. Per-session spike loading/binning and behavior loading/alignment dominate; full conversion also incurs process-pool and serialization costs.

ii.
```python
neural_dict, meta = load_session_neural(...)
binned_spikes = bin_spiking_data_current(...)
wheel_traces, wheel_mask = align_continuous_behavior(...)
```

iii. Timing measurements led the agent to replace remote loaders with local ALF reads and parallelize sessions; it estimated roughly 4.7 seconds per session sequentially.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-interval spike histogram loop, per-trial behavior interpolation loop, one-based block-counter loop, and output-assembly trial loop could be further vectorized or batched.

ii.
```python
for interval_idx, (t_beg, t_end) in enumerate(...): ...
for idx, (t_seg, v_seg) in enumerate(...): ...
for idx, value in enumerate(probability_left): ...
for trial_idx in range(session.kept_trial_count): ...
```

iii. The notes do not analyze these loops specifically; optimization focused instead on local I/O and session-level parallelism.

## 10-c. What processing does the code repeat multiple times?

i. It searches the filesystem repeatedly with recursive globs for each modality/probe, constructs interpolation objects once per trial, makes separate complete passes for wheel and whisker, and retries failed session processing from the beginning.

ii.
```python
matches = sorted(base_dir.glob(pattern))
fn = interp1d(t_seg, v_seg, ...)
wheel_traces = align_continuous_behavior(...)
whisker_traces = align_continuous_behavior(...)
```

iii. The agent only calls out repeated remote metadata/AP loading that its local implementation removed; it does not document the remaining repetitions.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/stores broad cluster QC metadata (`cluster_qc`, `cluster_good`) that are not used after session construction, retains timing/skipped-index fields only for logging, builds plot-only continuous traces, and loads the full release DataFrame even though only session/probe columns matter.

ii.
```python
"cluster_qc": {k: np.asarray(v) for k, v in clusters.to_dict("list").items()},
cluster_good=np.asarray(meta["good_clusters"]),
kept_trial_indices=keep_idx.astype(np.int64),
```

iii. The notes say it avoided unused behavioral streams and AP metadata, but do not acknowledge these remaining discarded intermediates.
