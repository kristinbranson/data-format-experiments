# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globbed every `sub-*/*.nwb` path, sorted the paths, opened each file with `NWBHDF5IO`, and processed it as one session.

ii.
```python
def iter_session_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
```

iii. The trajectory says the agent identified NWB as the canonical source and deliberately avoided loading the full units table as a dataframe for efficiency. It checked the 174-file inventory against the paper's session counts.

## 1-b. How are the data split into subjects?

i. Subject identity is read from `nwb.subject.subject_id` (falling back to the parent directory), and sessions are mapped to first-seen subject indices.

ii.
```python
subject_id = str(getattr(nwb.subject, "subject_id", path.parent.name.replace("sub-", "")))
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects); subjects.append(subj)
subject_idx.append(subject_to_idx[subj])
```

iii. The agent treated the NWB subject field as authoritative and reported 28 subjects after conversion.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session; retained records are appended in sorted path order.

ii.
```python
for i, path in enumerate(session_paths, 1):
    session_record, brain_region_to_idx = convert_session(path, brain_region_to_idx)
    neural.append(session_record["neural"])
```

iii. The agent inferred the file/session correspondence from the dataset layout. It dropped the one file with no usable good units, yielding 173 sessions.

## 1-d. How are the data split into trials?

i. Trial rows come from `nwb.trials`. The agent truncates to the minimum length of `is_good_trials` among retained units, slices go events to that count, and finally removes trials whose whole neural matrix is zero.

ii.
```python
trials_df = nwb.trials.to_dataframe()
n_recorded_trials = min(n_trials, int(min(recorded_trial_counts)))
trials_df = trials_df.iloc[:n_recorded_trials].copy()
go_times_all[:n_trials]
valid_trial_indices = [i for i, trial in enumerate(neural_trials) if np.any(trial)]
```

iii. A smoke test showed behavior continuing after spikes ended. The agent interpreted `is_good_trials` length as recording coverage, truncated late trials, then added the all-zero check to avoid behavior-only trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained unless beyond the inferred recorded prefix or all retained units have zero activity in the four-second window. Sessions with fewer than two surviving trials are dropped. There is no behavioral-performance filter and no explicit `free_water` filter.

ii.
```python
if n_recorded_trials < n_trials:
    trials_df = trials_df.iloc[:n_recorded_trials].copy()
valid_trial_indices = [i for i, trial in enumerate(neural_trials) if np.any(trial)]
if n_trials < 2:
    return None, brain_region_to_idx
```

iii. The agent rejected the paper's “regular trial” mask as analysis-specific because this task needs photostim, ignore, and early-lick labels. It justified its coverage filter from a session with 480 behavioral but only about 160 recorded trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units["spike_times"]`, restricted using `classification` and `anno_name`, and aligned using behavioral go-cue timestamps.

ii.
```python
spike_ends = np.asarray(units["spike_times"].data[:], dtype=np.int64)
spike_values = np.asarray(units["spike_times"].target.data[:], dtype=np.float64)
good_mask = (classifications == "good") & (anno_names != "")
```

iii. The agent found spike times to be the canonical neural representation and chose the classifier's `good` verdict rather than ad-hoc metric thresholds.

## 2-b. How is the `neural` data processed?

i. Spikes are assigned to a trial window and 50-ms bin, accumulated as counts, divided by 0.05 to produce Hz, and stored per trial as float16 neuron-by-time arrays. No smoothing or normalization is applied.

ii.
```python
bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
np.add.at(counts[unit_pos], (trial_idx[good], bin_idx[good]), 1)
trial_rates = counts[:, trial_idx, :].astype(np.float16) / np.float16(BIN_SIZE_S)
```

iii. The agent aimed to reproduce the reference firing-rate calculation while limiting the very large export's memory and disk footprint.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `classification == "good"` and a nonempty `anno_name`; sessions with no such unit are dropped.

ii.
```python
good_mask = (classifications == "good") & (anno_names != "")
good_unit_indices = np.flatnonzero(good_mask)
if len(good_unit_indices) == 0:
    return None, brain_region_to_idx
```

iii. The trajectory explicitly locks this rule and notes that it produces 69,453 labeled good units, close to the paper's reported scale; the label requirement also supports brain-region indexing.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Session-absolute spike times are shifted by each trial's go-cue timestamp and retained from -2.5 to +1.5 seconds.

ii.
```python
window_starts = go_times + WINDOW_START_S
window_ends = go_times + WINDOW_END_S
rel_spikes = spikes - go_times[trial_idx]
```

iii. The agent established that spikes and events share a clock and used the requested go-cue alignment directly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 nonoverlapping 50-ms bins over four seconds. Raw spike timestamps are histogrammed into these bins; no further rebinning occurs.

ii.
```python
BIN_SIZE_S = 0.05
N_BINS = int(round((WINDOW_END_S - WINDOW_START_S) / BIN_SIZE_S))
```

iii. These constants directly implement the decoder specification.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times`, go times, and trial start/stop bounds. Normally it selects the last sample onset between trial start and go.

ii.
```python
sample_starts = np.asarray(events["sample_start_times"].timestamps[:], dtype=np.float64)
lo = np.searchsorted(sample_starts, trial_start, side="left")
hi = np.searchsorted(sample_starts, go_time, side="right")
per_trial[i] = sample_starts[hi - 1]
```

iii. The agent recognized that early licking can replay the sample epoch and reasoned that the final tone before go is the behaviorally relevant one.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Tone onset is made relative to go, then subtracted from every go-relative bin center. Missing cases use the last sample in the trial or a median go-minus-sample fallback.

ii.
```python
sample_onsets_rel = sample_onsets - go_times
trial_input[0] = (BIN_CENTERS_S - sample_onsets_rel[trial_idx]).astype(np.float32)
per_trial[i] = go_times[i] - median_go_minus_sample
```

iii. The agent added fallbacks to tolerate malformed/missing event associations, while retaining the final pre-go sample rule in normal data.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at exactly the same 80 go-relative bin centers as neural activity.

ii.
```python
BIN_CENTERS_S = (BIN_EDGES_S[:-1] + BIN_EDGES_S[1:]) / 2.0
trial_input[0] = BIN_CENTERS_S - sample_onsets_rel[trial_idx]
```

iii. The shared grid was chosen to guarantee one input value per neural time bin.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial-table `photostim_onset`, `photostim_duration`, and `start_time`, together with the go timestamp.

ii.
```python
onset = maybe_float(trial.photostim_onset)
duration = maybe_float(trial.photostim_duration)
start_abs = float(trial.start_time) + onset
stim_starts[i] = start_abs - go_times[i]
```

iii. The agent observed that onset is trial-start-relative and that `N/A` denotes no stimulation.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. `N/A`, invalid, or NaN bounds produce all zeros. Otherwise, bins whose centers are in the half-open stimulation interval are 1.

ii.
```python
trial_input[1] = (
    (BIN_CENTERS_S >= stim_starts_rel[trial_idx])
    & (BIN_CENTERS_S < stim_ends_rel[trial_idx])
).astype(np.float32)
```

iii. The binary time series follows the requested time-varying input representation.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stim onset is converted to time relative to go, then compared with the same neural-bin centers.

ii.
```python
stim_starts[i] = start_abs - go_times[i]
(BIN_CENTERS_S >= stim_starts_rel[trial_idx])
```

iii. The shared go-relative clock and centers make bin indices correspond directly.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `left_lick_times`, `right_lick_times`, `go_start_times`, and `go_stop_times`, not from trial instruction/outcome fields.

ii.
```python
left_licks = np.asarray(events["left_lick_times"].timestamps[:], dtype=np.float64)
right_licks = np.asarray(events["right_lick_times"].timestamps[:], dtype=np.float64)
```

iii. The agent intentionally defined choice as the first physical response-window lick, considering it more direct than reconstructing choice from instruction and outcome.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For each trial, the first left and right lick at or after go is found; candidates at or after response end are discarded. The earlier side is coded 0/1, otherwise no lick is 2, and the value is repeated over 80 bins.

ii.
```python
if left_time < right_time: choices[i] = 0
elif right_time < left_time: choices[i] = 1
else: choices[i] = 2
trial_output[0] = choices[trial_idx]
```

iii. The trajectory states this first-response-window-lick choice explicitly; repeating it produces a uniform output shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials table's `outcome` column.

ii.
```python
outcome_strings = trials_df["outcome"].astype(str).to_numpy()
```

iii. The requested three categories already exist in the raw table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped `ignore:0`, `miss:1`, `hit:2` and repeated across bins.

ii.
```python
OUTCOME_TO_INT = {"ignore": 0, "miss": 1, "hit": 2}
trial_output[1] = outcome_labels[trial_idx]
```

iii. This is a direct categorical encoding in the requested order.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials table's `early_lick` column.

ii.
```python
early_strings = trials_df["early_lick"].astype(str).to_numpy()
```

iii. The table already supplies the requested trial-level label.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and `early` to 1; the result is repeated across bins.

ii.
```python
EARLY_TO_INT = {"no early": 0, "early": 1}
trial_output[2] = early_labels[trial_idx]
```

iii. This directly implements the two requested categories.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and columns 1 (y) and 2 (tracking likelihood) from `Camera0_side_TongueTracking`.

ii.
```python
ts = nwb.acquisition["BehavioralTimeSeries"].time_series["Camera0_side_TongueTracking"]
y = data[:, 1]
likelihood = data[:, 2]
```

iii. The agent identified this as the canonical side-camera tongue stream and used confidence to represent visibility.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood at least 0.9 define visibility and raw visible-frame 40th/60th percentiles. For each neural bin, only the final camera frame before the bin end is sampled; it is accepted if it falls within the bin and passes confidence. Otherwise the bin remains not visible.

ii.
```python
visible = likelihood >= likelihood_threshold
q40, q60 = np.quantile(y[visible], [0.4, 0.6])
frame_idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
like_valid = likelihood[idx] >= likelihood_threshold
```

iii. The trajectory says the agent chose session-wide visible-frame percentiles using DLC confidence. It did not document why it sampled one frame rather than averaging all visible frames in a bin.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Session-wide q40/q60 cutoffs from visible raw frames produce 0 below q40, 1 inclusively between q40 and q60, 2 above q60, and 3 for absent/low-confidence frames. If no frame is visible, quantiles fall back to all y values.

ii.
```python
labels = np.full(N_BINS, 3, dtype=np.int8)
labels[visible_bins[y_vis < q40]] = 0
labels[visible_bins[(y_vis >= q40) & (y_vis <= q60)]] = 1
labels[visible_bins[y_vis > q60]] = 2
```

iii. The percentiles and per-session scope come from the task; the agent chose 0.9 as a conservative visibility threshold.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each go cue, absolute starts and ends for the same 80 bins are formed. The last frame strictly before each end is used if its timestamp is at least the bin start.

ii.
```python
bin_starts = go_time + BIN_EDGES_S[:-1]
bin_ends = go_time + BIN_EDGES_S[1:]
frame_idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
```

iii. Camera, spike, and event timestamps share the NWB session clock, so no interpolation or clock correction was considered necessary.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. `N/A`/NaN photostim becomes no stimulation; missing tone associations use within-trial or median timing fallbacks; missing tongue visibility becomes category 3; sessions with no labeled good units and sessions/trials without usable neural data are dropped. Per-session exceptions are caught, logged, and skipped.

ii.
```python
except Exception as exc:
    skipped_sessions.append({"session_path": str(path), "reason": repr(exc)})
if np.isnan(stim_starts_rel[trial_idx]): trial_input[1] = 0.0
labels = np.full(N_BINS, 3, dtype=np.int8)
```

iii. The agent sought a complete robust export and used explicit missing categories or exclusion rather than fabricating neural activity, though broad exception swallowing can hide unexpected corruption.

## 10-a. What are the most time-consuming steps of the code?

i. Reading large NWB spike/camera arrays, the per-unit spike assignment, building dense neural matrices, writing the 5.6-GB pickle, and downstream training dominate.

ii.
```python
for unit_pos, unit_idx in enumerate(good_unit_indices):
    ...
    np.add.at(counts[unit_pos], (trial_idx[good], bin_idx[good]), 1)
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory explicitly optimized NWB access, monitored full-export runtime/memory, and reported that conversion was fast enough while CPU decoder training was the long bounded step.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Loops over trials for sample lookup, choice, photostim, tongue extraction, input/output construction, trial-list conversion, region mapping, and some of the per-unit spike work could be vectorized or batched.

ii.
```python
for i, (trial_start, go_time) in enumerate(...): ...
for go_time in go_times: ...
for trial_idx in range(n_trials): ...
for i, region in enumerate(unit_regions): ...
```

iii. No explicit trajectory justification was given for most loops; helper-based clarity and bounded per-session processing appear to have been prioritized.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly scans event arrays per trial, repeatedly constructs bin boundaries, loops over trials to form several outputs, calls `iter_session_paths` again for metadata, and computes control-performance statistics that duplicate trial-column conversions.

ii.
```python
for go_time in go_times:
    bin_starts = go_time + BIN_EDGES_S[:-1]
...
"n_source_sessions": len(iter_session_paths(data_dir)),
performance, left_hits, right_hits, n_regular = compute_control_performance(trials_df)
```

iii. The agent did not discuss these repetitions; they are mostly modest relative to spike I/O and dense-array construction.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes control performance, side-specific hit counts, regular-trial counts, tongue quantiles, detailed skipped-session records, and time-bin-center metadata; these do not feed the decoder. It also constructs intermediate dataframes and lists solely for filtering/metadata.

ii.
```python
performance, left_hits, right_hits, n_regular = compute_control_performance(trials_df)
"control_hit_left": left_hits,
"tongue_visible_q40": tongue_quantiles[0],
"skipped_sessions": skipped_sessions,
```

iii. The trajectory used counts and performance-style checks for sanity and provenance, but did not claim these metadata were required by downstream analysis.
