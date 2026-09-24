# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the first available local release manifest, constructs each session directory from its lab/subject/date/number fields, skips manifest entries whose directory is absent, and recursively selects the latest matching revision of each required file. It processes sessions in up to eight threads. Unlike the reference, it does not use ONE, release-cache searches, or `SessionLoader`/`SpikeSortingLoader`.

ii.
```python
manifest = load_session_manifest()
for eid, row in manifest.iterrows():
    session_path = DATA_ROOT / row["lab"] / "Subjects" / row["subject"] / str(row["date"]) / f"{int(row['number']):03d}"
    if not session_path.exists():
        missing.append(eid); continue
    specs.append(SessionSpec(eid=eid, subject=str(row["subject"]), session_path=session_path, ...))
```

iii. The notes say the 2025 manifest is the canonical 459-session release and local availability should determine the usable subset. Direct local loading and memory mapping were chosen for reproducibility and speed.

## 1-b. How are the data split into subjects?

i. Subject IDs come directly from the manifest. The final subject vocabulary is the sorted set of subjects among retained sessions, and `subject_idx` maps every retained session into it.

ii.
```python
subject=str(row["subject"])
subjects = sorted({s.subject for s in sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
"subject_idx": np.array([subject_to_idx[s.subject] for s in sessions], dtype=np.int16)
```

iii. The AI states that session subject metadata should supply `subjects` and `subject_idx`, with deterministic ordering.

## 1-c. How are the data split into sessions?

i. Each manifest row/EID is treated as one session, resolved to one local session directory, processed independently, and appended as one element of each top-level session list.

ii.
```python
for eid, row in manifest.iterrows():
    ... specs.append(SessionSpec(eid=eid, ...))
...
data["neural"].append([trial.astype(np.float16) for trial in session.neural])
```

iii. The notes identify sessions as the unit of analysis and merge probes within a session because they share behavior.

## 1-d. How are the data split into trials?

i. The trials parquet already has one row per trial. Retained rows provide `stimOn_times`; spike and behavior streams are sliced into `[-0.5, 1.5]` s windows around each onset.

ii.
```python
align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
intervals = np.c_[align_times + window[0], align_times + window[1]]
for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
    ... results.append(trial_counts)
```

iii. The AI says the raw trials table defines trial boundaries and follows the repository's stimulus-aligned two-second windows.

## 1-e. How are trials filtered based on quality controls?

i. It requires RT in `[0.08,2]` s, trial duration from go cue to feedback at most 10 s, nonzero choice, and nonmissing required event/task columns. After alignment it additionally requires complete wheel and whisker coverage and at least one spike in the trial. Sessions with fewer than two surviving trials are dropped.

ii.
```python
mask &= rt >= TRIAL_MASK_RT[0]; mask &= rt <= TRIAL_MASK_RT[1]
mask &= (trials["feedback_times"] - trials["goCue_times"]) <= MAX_TRIAL_LEN
mask &= trials["choice"] != 0
for col in required: mask &= trials[col].notna().to_numpy()
...
neural_mask = np.array([np.any(trial) for trial in neural_trials], dtype=bool)
combined_mask = wheel_mask & whisk_mask & neural_mask
```

iii. The notes attribute the event, RT, no-choice, and 10 s filters to `load_trials_and_mask`; behavior coverage follows the reference alignment concept. The all-zero neural exclusion is documented in the README but not supported by the human reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays derive from `spikes.times.npy` and `spikes.clusters.npy`; cluster quality comes from `clusters.metrics.pqt`, and regions from cluster-channel indices plus `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
spikes_times_file = pick_one_file(pykilo_path, "spikes.times.npy")
spikes_clusters_file = pick_one_file(pykilo_path, "spikes.clusters.npy")
metrics_file = pick_one_file(pykilo_path, "clusters.metrics.pqt")
clusters_channels_file = pick_one_file(pykilo_path, "clusters.channels.npy")
channels_ids_file = pick_one_file(pykilo_path, "channels.brainLocationIds_ccf_2017.npy")
```

iii. The mapping plan says spike time/cluster arrays from all probes form neural activity, while metrics and anatomy curate and annotate neurons.

## 2-b. How is the `neural` data processed?

i. Good-cluster spikes from all probes are reindexed, concatenated, sorted, and counted in 20 ms bins per trial with `bincount2D`. The saved values are raw spike counts in `float16`; unlike the human reference, they are not divided by 0.02 to obtain Hz.

ii.
```python
counts, _, cluster_idx = bincount2D(..., xbin=binsize, xlim=[start, end])
trial_counts[cluster_idx, : counts.shape[1]] = counts.astype(np.float16)
...
data["neural"].append([trial.astype(np.float16) for trial in session.neural])
```

iii. The notes explicitly plan “raw binned spike counts, not standardized z-scores,” but do not justify departing from the human conversion's firing-rate scaling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It keeps clusters with `metrics.label >= 1`. It assigns raw Allen acronyms from channel IDs and neither maps them to Beryl nor removes `void`; trials with no spikes across all kept units/bins are also removed.

ii.
```python
good_mask = cluster_labels >= label_threshold
spike_keep = good_mask[spikes_clusters]
cluster_regions = br.id2acronym(cluster_region_ids)
...
neural_mask = np.array([np.any(trial) for trial in neural_trials], dtype=bool)
```

iii. The AI chose the stringent label because it reproduces the paper's well-isolated-unit count and makes the dense export tractable. It chose raw acronyms to preserve atlas detail.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial interval is `[stimOn_times-0.5, stimOn_times+1.5)`, and absolute spike timestamps are binned within that interval.

ii.
```python
align_times = masked_trials["stimOn_times"].to_numpy(dtype=np.float64)
intervals = np.c_[align_times + window[0], align_times + window[1]]
idx_starts = np.searchsorted(spike_times, intervals[:, 0], side="left")
idx_ends = np.searchsorted(spike_times, intervals[:, 1], side="left")
```

iii. The notes prioritize the executable repository settings and the user's explicit stimulus-onset alignment over conflicting prose descriptions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The code bins continuous spike times directly into 100 nonoverlapping 20 ms bins over two seconds. There is no further rebinning or smoothing.

ii.
```python
BINSIZE_S = 0.02
N_BINS = int(round((TIME_WINDOW[1] - TIME_WINDOW[0]) / BINSIZE_S))
counts, _, cluster_idx = bincount2D(..., xbin=binsize, ...)
```

iii. The AI says 20 ms and 100 steps match the executable code and generic model description.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is generated from the fixed window and bin size associated with `stimOn_times`, rather than read as a separate raw variable.

ii.
```python
TIME_WINDOW = (-0.5, 1.5)
BINSIZE_S = 0.02
def make_time_input():
    return np.linspace(TIME_WINDOW[0] + BINSIZE_S, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

iii. The notes describe it as the shared signed stimulus-aligned grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. It creates 100 equally spaced right-edge values from -0.48 through 1.50 s and copies the vector to every trial. The reference instead uses bin centers from -0.49 through 1.49 s.

ii.
```python
time_input = make_time_input()
input_trial = np.vstack([time_input, np.full(N_BINS, block_num, dtype=np.float32)])
```

iii. The AI deliberately chose the repository behavior interpolation's right-edge sample grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Both arrays have 100 positions and the behavior outputs use the same right-edge query grid. However, neural values summarize bins `[edge_i, edge_{i+1})` while the time value labels their right edge, not their center.

ii.
```python
x_rel = np.linspace(window[0] + binsize, window[1], n_bins)
...
return np.linspace(TIME_WINDOW[0] + BINSIZE_S, TIME_WINDOW[1], N_BINS, dtype=np.float32)
```

iii. The AI considered the common 20 ms grid sufficient for bin-for-bin alignment.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from consecutive values of the raw `probabilityLeft` column; a changed value begins a new block.

ii.
```python
block_trial_number = compute_trial_number_in_block(trials["probabilityLeft"].to_numpy())
```

iii. The notes explain that no explicit block ID exists, so changes in the block prior define boundaries.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A loop counts from 1 within each block and resets to 1 on a prior change. It is calculated before filtering so removed trials still advance the count, then broadcast over 100 bins. The human reference counts from 0.

ii.
```python
count = 1; counters[0] = count
for i in range(1, len(prob_left)):
    if prob_left[i] == prob_left[i - 1]: count += 1
    else: count = 1
    counters[i] = count
```

iii. The AI correctly justified computing on the original trial sequence, but did not justify its one-based departure.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from the trials table's `choice` column.

ii.
```python
choice_vals = map_choice_to_binary(masked_keep["choice"].to_numpy(dtype=np.float64))
```

iii. The notes use IBL's raw sign convention and exclude no-response trials.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Raw `+1` maps to left/0 and `-1` to right/1; unexpected values raise an error. The value is repeated across all 100 time bins.

ii.
```python
mapped[choice_values == 1] = 0
mapped[choice_values == -1] = 1
choice.append(np.full(N_BINS, choice_val, dtype=np.int16))
```

iii. This is the explicit target mapping and gives static variables consistent `(d,T)` shapes.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes directly from `trials.probabilityLeft`.

ii.
```python
prior_vals = map_prior_to_categorical(masked_keep["probabilityLeft"].to_numpy(dtype=np.float64))
```

iii. The AI identifies this column as the block prior required by the task.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values close to 0.2, 0.5, and 0.8 map to 0, 1, and 2, respectively; unknown values error. Each result is repeated over time.

ii.
```python
mapped[np.isclose(prob_left, 0.2)] = 0
mapped[np.isclose(prob_left, 0.5)] = 1
mapped[np.isclose(prob_left, 0.8)] = 2
prior.append(np.full(N_BINS, prior_val, dtype=np.int16))
```

iii. The mapping is explicitly required by the user.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It derives from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
pos = np.asarray(np.load(wheel_pos_file), dtype=np.float64)
ts = np.asarray(np.load(wheel_ts_file), dtype=np.float64)
pos_interp, ts_interp = interpolate_position(ts, pos, freq=1000)
vel, _ = velocity_filtered(pos_interp, 1000)
return ts_interp, np.abs(vel)
```

iii. The notes say this matches the bundled wheel utilities and reference `wheel-speed` definition.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is interpolated at 1 kHz, filtered/differentiated into velocity, converted to absolute speed, finite samples retained, and linearly interpolated onto each trial's 100 query times.

ii.
```python
valid_source = np.isfinite(target_times) & np.isfinite(target_values)
interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. The AI follows the recommended IBL wheel processing and the repository's linear behavior interpolation.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Two thresholds are the 1/3 and 2/3 quantiles pooled across every valid sample in every retained session. `np.digitize` creates classes 0–2. Degenerate thresholds fall back to equal-range cuts. The reference uses separate percentiles within each session.

ii.
```python
q1, q2 = np.quantile(concat, [1 / 3, 2 / 3])
...
return np.digitize(values, bins=np.array(edges, dtype=np.float32), right=False).astype(np.int16)
```

iii. The AI chose global tertiles to give categories a common dataset-wide meaning and improve overall class balance.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated at stimulus onset plus `[-0.48,-0.46,...,1.50]`, yielding 100 values corresponding by index to the 100 neural bins, though these are bin right edges rather than the reference's centers.

ii.
```python
x_rel = np.linspace(window[0] + binsize, window[1], n_bins)
outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. The notes say all streams must share the requested stimulus-onset grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `leftCamera.ROIMotionEnergy.npy` and matching camera times if available, otherwise the right-camera equivalents.

ii.
```python
for camera in ("left", "right"):
    me_file = pick_one_file(session_path / "alf", f"{camera}Camera.ROIMotionEnergy.npy")
    times_file = pick_one_file(session_path / "alf", f"*{camera}Camera.times.npy")
```

iii. The notes say left-first/right-fallback matches the reference and retains more sessions.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. If time/value lengths differ, both are truncated to the shorter length. Nonfinite samples are removed, and the released trace is linearly interpolated to the 100 trial query times; there is no filtering or normalization.

ii.
```python
if len(values) != len(times):
    n = min(len(values), len(times)); values = values[:n]; times = times[:n]
...
interp = interp1d(ts, vals, kind="linear", fill_value="extrapolate")
```

iii. The AI states the released motion-energy signal should be used directly and interpolated like other continuous behavior.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses dataset-wide pooled 1/3 and 2/3 quantiles and `np.digitize`, with the same degeneracy fallback as wheel speed. The reference instead computes session-wise percentiles.

ii.
```python
whisker_edges = compute_tertile_edges(
    trial_values for session in processed for trial_values in session.whisker_cont)
discretize_three_bins(whisk_cont, whisker_edges)
```

iii. The stated rationale is common global category meaning and balanced classes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera motion energy is evaluated at stimulus onset plus the same 100 right-edge query times used for wheel and the time input. This is index-aligned but shifted 10 ms later than the reference bin-center convention.

ii.
```python
x_rel = np.linspace(window[0] + binsize, window[1], n_bins)
outputs.append(interp(align_time + x_rel).astype(np.float32))
```

iii. The AI treats shared stimulus clock and query grid as the required alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing session directories are skipped; missing trials files error; sessions missing spike assets or wheel/whisker files are skipped through raised exceptions at session level; no-good-unit and fewer-than-two-trial sessions are dropped. Required trial NaNs are masked, nonfinite behavior samples removed, mismatched camera arrays truncated, incomplete behavior windows rejected, and unmappable categorical values raise errors.

ii.
```python
if not session_path.exists(): missing.append(eid); continue
if any(x is None for x in required): raise FileNotFoundError(...)
valid_source = np.isfinite(target_times) & np.isfinite(target_values)
if len(values) != len(times):
    n = min(len(values), len(times)); values = values[:n]; times = times[:n]
```

iii. The AI's plan is to derive a usable local subset, reject trials without full aligned data, and skip sessions that cannot meet the decoder format.

## 10-a. What are the most time-consuming steps of the code?

i. Loading/filtering large spike arrays and trial-by-trial spike binning are the main costs; behavior interpolation also loops over trials. Sessions are threaded to reduce elapsed time.

ii.
```python
spikes_times = np.load(spikes_times_file, mmap_mode="r")
...
for idx0, idx1, (start, end) in zip(idx_starts, idx_ends, intervals):
    counts, _, cluster_idx = bincount2D(...)
```

iii. The notes explicitly call trial-by-trial spike binning the hot path and use memory mapping plus up to eight workers.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike binning, behavior interpolation, block numbering, trial input/output construction, session assembly, and per-trial discretization all use Python loops. The first two are the most plausible performance targets; block counting can also be vectorized cheaply.

ii.
```python
for idx0, idx1, (start, end) in zip(...): ...
for i, align_time in enumerate(align_times): ...
for i in range(1, len(prob_left)): ...
for block_num, choice_val, prior_val in zip(...): ...
```

iii. The AI identified trial spike binning as the primary inefficiency, while retaining per-trial loops for variable-length slices and shared-grid interpolation.

## 10-c. What processing does the code repeat multiple times?

i. It recursively searches for several files per probe/session; constructs a new `BrainRegions` in every worker call; loops over each trial once for spikes, once for each behavior, once to build static arrays, and again during final output discretization. Continuous wheel/whisker traces are retained and traversed again after global thresholds are known.

ii.
```python
def process_session_worker(spec):
    return process_session(spec, BrainRegions())
...
wheel_edges = compute_tertile_edges(...)
...
discretize_three_bins(wheel_cont, wheel_edges)
```

iii. Global thresholds require a second pass over stored behavior; the notes do not otherwise discuss repeated processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/counts all trial spikes before removing all-zero trials; stores continuous behavior solely to compute thresholds and later discards it; records diagnostic fields such as raw counts, kept indices, camera source, and totals that are absent from the final pickle; maps detailed raw atlas regions although decoding does not otherwise use their hierarchy. Plotting is optional and only occurs when requested.

ii.
```python
neural_trials = bin_spikes_for_trials(...)
neural_mask = np.array([np.any(trial) for trial in neural_trials], dtype=bool)
...
wheel_cont=[np.asarray(x, dtype=np.float32) for x in wheel_keep]
...
kept_trial_indices=..., n_clusters_total=..., raw_n_trials=...
```

iii. These intermediates support QC, global discretization, summaries, and optional plots, but the notes do not claim they are needed by downstream training.
