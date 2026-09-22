# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files stored under `/app/data/sub-*/`. It uses `pynwb.NWBHDF5IO` to open each `.nwb` file and reads trials from `nwb.trials.to_dataframe()`, units from `nwb.units`, and behavioral events from `nwb.acquisition['BehavioralEvents']`. All session paths are discovered via a sorted glob over `sub-*/*.nwb`.

ii.
```python
def iter_session_paths(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))
```

```python
with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
    nwb = io.read()
    trials_df = nwb.trials.to_dataframe()
    units = nwb.units
```

iii. The agent recognized that the NWB files are the canonical data format on disk and adapted the reference code's logic (originally designed for DataJoint `.mat` exports) to read from NWB's native structure. The agent stated: "I've verified the NWB files expose trials and units directly, not just the old `.mat` export."

## 1-b. How are the data split into subjects?

i. Subject IDs are extracted from `nwb.subject.subject_id` for each session, with a fallback to the directory name (`sub-XXXXX` -> `XXXXX`). Unique subjects are collected as sessions are processed and assigned sequential indices.

ii.
```python
subject_id = str(getattr(nwb.subject, "subject_id", path.parent.name.replace("sub-", "")))
```

```python
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
```

iii. The agent inspected the NWB metadata and found `subject_id` as a numeric string (e.g., `'440956'`). The fallback to directory name was a defensive measure in case the attribute is missing.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Of 174 NWB files, 173 are kept; one is dropped because it has zero "good" units with nonempty histology labels. Sessions are processed in sorted file order.

ii.
```python
session_paths = iter_session_paths(data_dir)
# ...
for i, path in enumerate(session_paths, 1):
    session_record, brain_region_to_idx = convert_session(path, brain_region_to_idx)
```

iii. The agent investigated the discrepancy between 174 files and the paper's "173 sessions" and found one session had no qualifying units. It explicitly tested whether the paper's behavioral performance filter (>65% correct) should be applied and decided against it: "The published behavior-quality filter keeps only 145 of 174 sessions, so that criterion was clearly analysis-specific rather than the dataset-wide inclusion rule."

## 1-d. How are the data split into trials?

i. Trials come from `nwb.trials.to_dataframe()`, one row per behavioral trial. Go cue times come from `BehavioralEvents/go_start_times`. The agent truncates the trial table to only the trials covered by neural recording, using the `units['is_good_trials']` field to determine the recorded trial count.

ii.
```python
trials_df = nwb.trials.to_dataframe()
n_trials = len(trials_df)
# ...
go_times, response_ends = get_go_times_and_response_ends(nwb, n_trials)
```

```python
recorded_trial_counts = [len(np.asarray(units["is_good_trials"][u])) for u in good_unit_indices]
n_recorded_trials = int(min(recorded_trial_counts)) if recorded_trial_counts else n_trials
n_recorded_trials = min(n_recorded_trials, n_trials)
if n_recorded_trials < n_trials:
    trials_df = trials_df.iloc[:n_recorded_trials].copy()
```

iii. The agent discovered that some sessions have more behavioral trials than neural recording coverage (e.g., one session had 480 behavioral trials but spikes ended around trial 158). Rather than using `obs_intervals` (as the reference does), the agent used `is_good_trials` length to determine the cut-off and simply truncated from the end.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials only by neural recording coverage — sessions are truncated to the number of trials covered by neural recording (using `is_good_trials`). After neural binning, trials where all spike counts are zero are also removed. There is NO filtering of `free_water` trials. Sessions with fewer than 2 trials are dropped.

ii.
```python
# Truncation to recorded trials
n_recorded_trials = int(min(recorded_trial_counts))
if n_recorded_trials < n_trials:
    trials_df = trials_df.iloc[:n_recorded_trials].copy()

# Post-hoc removal of all-zero neural trials
valid_trial_indices = [i for i, trial in enumerate(neural_trials) if np.any(trial)]
if len(valid_trial_indices) < len(neural_trials):
    neural_trials = [neural_trials[i] for i in valid_trial_indices]
    input_trials = [input_trials[i] for i in valid_trial_indices]
    output_trials = [output_trials[i] for i in valid_trial_indices]
```

iii. The agent chose to keep all trial types (including early-lick, auto-water, free-water, photostimulation, and no-response trials) because they are needed as decoder outputs/inputs. The agent did not use `obs_intervals` for trial filtering and instead relied on `is_good_trials` length to trim unrecorded trials, plus a post-hoc check removing trials with all-zero neural data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` — the sorted spike times of each unit in session-absolute seconds. Only units passing the quality filter contribute. Go cue times (`BehavioralEvents/go_start_times`) define the trial windows.

ii.
```python
spike_ends = np.asarray(units["spike_times"].data[:], dtype=np.int64)
spike_values = np.asarray(units["spike_times"].target.data[:], dtype=np.float64)
# ...
for unit_pos, unit_idx in enumerate(good_unit_indices):
    start = 0 if unit_idx == 0 else spike_ends[unit_idx - 1]
    end = spike_ends[unit_idx]
    spikes = spike_values[start:end]
```

iii. The agent accesses spike times via the raw HDF5 index structure rather than `units.to_dataframe()`, because the latter "may silently drag in ragged spike-time columns and make the conversion much slower."

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50ms bins spanning -2.5s to +1.5s relative to go cue onset. Spike counts per bin are divided by `BIN_SIZE_S` (0.05s) to convert to firing rates in Hz. The result is stored as `float16`.

ii.
```python
rel_spikes = spikes - go_times[trial_idx]
bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
good = (bin_idx >= 0) & (bin_idx < N_BINS)
if np.any(good):
    np.add.at(counts[unit_pos], (trial_idx[good], bin_idx[good]), 1)
# ...
trial_rates = counts[:, trial_idx, :].astype(np.float16) / np.float16(BIN_SIZE_S)
```

iii. The agent uses standard histogram binning (non-overlapping bins) matching the 50ms bin width from the instructions. No smoothing, normalization, or baseline subtraction is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are kept if `classification == 'good'` AND `anno_name != ''` (nonempty histology annotation). Sessions with zero qualifying units are dropped entirely.

ii.
```python
classifications = get_vector_strings(units, "classification")
anno_names = get_vector_strings(units, "anno_name")
good_mask = (classifications == "good") & (anno_names != "")
good_unit_indices = np.flatnonzero(good_mask)
if len(good_unit_indices) == 0:
    return None, brain_region_to_idx
```

iii. The agent uses both the QC classifier verdict (`classification`) and the presence of histology annotation (`anno_name`) as filters. The `anno_name` filter ensures that brain region information exists for each kept unit, which is needed for the `brain_region_idx` field.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times and go cue times are on the same session-absolute clock. The agent computes relative spike times by subtracting the go cue time, then assigns spikes to bins based on their offset from the window start.

ii.
```python
window_starts = go_times + WINDOW_START_S
window_ends = go_times + WINDOW_END_S
# ...
rel_spikes = spikes - go_times[trial_idx]
bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
```

iii. Alignment is straightforward since NWB timestamps share a global clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 50ms, producing 80 bins spanning -2.5s to +1.5s relative to go cue onset. Bin edges are computed with `np.linspace`. No rebinning is applied — spikes are binned directly from their raw timestamps.

ii.
```python
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
N_BINS = int(round((WINDOW_END_S - WINDOW_START_S) / BIN_SIZE_S))
BIN_EDGES_S = np.linspace(WINDOW_START_S, WINDOW_END_S, N_BINS + 1, dtype=np.float64)
BIN_CENTERS_S = (BIN_EDGES_S[:-1] + BIN_EDGES_S[1:]) / 2.0
```

iii. The 50ms bin width and -2.5 to +1.5s window are specified in the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` in `BehavioralEvents`, together with the go cue time of each trial. For each trial, the last `sample_start_times` timestamp before the go cue (but within the trial bounds) is selected as the tone onset.

ii.
```python
def get_sample_onsets(nwb, trial_starts, go_times, trial_stops):
    events = nwb.acquisition["BehavioralEvents"].time_series
    sample_starts = np.asarray(events["sample_start_times"].timestamps[:], dtype=np.float64)
    per_trial = np.full(len(go_times), np.nan, dtype=np.float64)
    for i, (trial_start, go_time) in enumerate(zip(trial_starts, go_times)):
        lo = np.searchsorted(sample_starts, trial_start, side="left")
        hi = np.searchsorted(sample_starts, go_time, side="right")
        if hi > lo:
            per_trial[i] = sample_starts[hi - 1]
```

iii. The agent recognized that early-lick trials replay the sample epoch, so a trial can have multiple tone onsets: "tone onset is not a single NWB trial column because early-lick trials can replay sample epochs... the correct per-trial tone reference is the final sample onset before the go cue."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone onset is converted to go-cue-relative time, then the input is computed as `BIN_CENTERS_S - sample_onset_relative_to_go`. There is also a fallback: if no sample onset is found before the go cue within the trial, it searches the full trial interval; if still not found, it uses the session median go-to-sample offset.

ii.
```python
sample_onsets_rel = sample_onsets - go_times
# ...
trial_input[0] = (BIN_CENTERS_S - sample_onsets_rel[trial_idx]).astype(np.float32)
```

Fallback:
```python
if np.any(valid):
    median_go_minus_sample = float(np.median(go_times[valid] - per_trial[valid]))
else:
    median_go_minus_sample = 1.85
for i in np.where(~valid)[0]:
    # ... try wider search, then fallback
    per_trial[i] = go_times[i] - median_go_minus_sample
```

iii. The fallback logic handles edge cases where sample onset timing is missing.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin centers defined relative to the go cue, so alignment is inherent. The time-from-tone input is computed at the same 80 bin centers as the neural data.

ii.
```python
trial_input[0] = (BIN_CENTERS_S - sample_onsets_rel[trial_idx]).astype(np.float32)
```

iii. N/A — alignment is guaranteed by using the same go-cue-relative time grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, plus `start_time` to convert to absolute time and `go_times` to convert to go-cue-relative time.

ii.
```python
def get_photostim_relative_intervals(trials_df, go_times):
    for i, trial in enumerate(trials_df.itertuples()):
        onset = maybe_float(trial.photostim_onset)
        duration = maybe_float(trial.photostim_duration)
        if onset is None or duration is None:
            continue
        start_abs = float(trial.start_time) + onset
        end_abs = start_abs + duration
        stim_starts[i] = start_abs - go_times[i]
        stim_ends[i] = end_abs - go_times[i]
```

iii. The onsets are stored relative to trial start and need to be converted to go-cue-relative times.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time-varying signal is created: each bin is 1.0 if its center falls within the stimulation interval [onset, offset), and 0.0 otherwise. Trials without photostimulation get all zeros.

ii.
```python
if np.isnan(stim_starts_rel[trial_idx]) or np.isnan(stim_ends_rel[trial_idx]):
    trial_input[1] = 0.0
else:
    trial_input[1] = (
        (BIN_CENTERS_S >= stim_starts_rel[trial_idx]) & (BIN_CENTERS_S < stim_ends_rel[trial_idx])
    ).astype(np.float32)
```

iii. The binary representation follows the instructions' requirement for representing time-based inputs.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The stimulation onset/offset are expressed relative to the go cue (same reference as neural bins), so comparison with bin centers provides alignment.

ii.
```python
stim_starts[i] = start_abs - go_times[i]
stim_ends[i] = end_abs - go_times[i]
# ...
(BIN_CENTERS_S >= stim_starts_rel[trial_idx]) & (BIN_CENTERS_S < stim_ends_rel[trial_idx])
```

iii. N/A

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from the `left_lick_times` and `right_lick_times` event streams in `BehavioralEvents`, plus `go_start_times` and `go_stop_times` to define the response window.

ii.
```python
def get_first_response_choices(nwb, go_times, response_ends):
    events = nwb.acquisition["BehavioralEvents"].time_series
    left_licks = np.asarray(events["left_lick_times"].timestamps[:], dtype=np.float64)
    right_licks = np.asarray(events["right_lick_times"].timestamps[:], dtype=np.float64)
    # ...
    if left_time < right_time:
        choices[i] = CHOICE_TO_INT["left"]
    elif right_time < left_time:
        choices[i] = CHOICE_TO_INT["right"]
    else:
        choices[i] = CHOICE_TO_INT["no lick"]
```

iii. The agent derived choice from the first lick in the response window rather than from `trial_instruction x outcome`, reasoning that this directly captures what the animal actually did.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For each trial, the first left and right lick times after the go cue (but before response end) are found. The earlier lick determines the choice. If neither occurs within the response window, choice is "no lick" (2). The per-trial value is broadcast across all 80 time bins.

ii.
```python
left_idx = np.searchsorted(left_licks, go_time, side="left")
right_idx = np.searchsorted(right_licks, go_time, side="left")
left_time = left_licks[left_idx] if left_idx < len(left_licks) else np.inf
right_time = right_licks[right_idx] if right_idx < len(right_licks) else np.inf
if left_time >= response_end:
    left_time = np.inf
if right_time >= response_end:
    right_time = np.inf
```

```python
trial_output[0] = choices[trial_idx]  # broadcast across bins
```

iii. The values are `["left", "right", "no lick"]` mapped to integers 0, 1, 2.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which contains strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcome_strings = trials_df["outcome"].astype(str).to_numpy()
outcome_labels = np.asarray([OUTCOME_TO_INT[x] for x in outcome_strings], dtype=np.int8)
```

iii. The trials table stores outcome explicitly with the three categories required.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to integers: `ignore` -> 0, `miss` -> 1, `hit` -> 2. The per-trial value is broadcast across all 80 bins.

ii.
```python
OUTCOME_TO_INT = {"ignore": 0, "miss": 1, "hit": 2}
# ...
trial_output[1] = outcome_labels[trial_idx]
```

iii. Straightforward mapping matching the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds `'no early'` and `'early'`.

ii.
```python
early_strings = trials_df["early_lick"].astype(str).to_numpy()
early_labels = np.asarray([EARLY_TO_INT[x] for x in early_strings], dtype=np.int8)
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The two strings are mapped to integers: `no early` -> 0, `early` -> 1. The per-trial value is broadcast across all 80 bins.

ii.
```python
EARLY_TO_INT = {"no early": 0, "early": 1}
# ...
trial_output[2] = early_labels[trial_idx]
```

iii. Direct mapping matching the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `Camera0_side_TongueTracking` in `nwb.acquisition['BehavioralTimeSeries']`, which is a `(n_frames, 3)` array of `(tongue_x, tongue_y, tongue_likelihood)` with timestamps.

ii.
```python
ts = nwb.acquisition["BehavioralTimeSeries"].time_series["Camera0_side_TongueTracking"]
timestamps = np.asarray(ts.timestamps[:], dtype=np.float64)
data = np.asarray(ts.data[:], dtype=np.float32)
y = data[:, 1]
likelihood = data[:, 2]
```

iii. This is the only tongue measurement in the NWB files.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with `likelihood >= 0.9` are considered visible. The 40th and 60th percentiles of y-position are computed over **raw visible frames** (not bin means) across the whole session. Per-trial bins are assigned by taking the last video frame before each bin's end time and classifying its y-value: 0 if y < q40, 1 if q40 <= y <= q60, 2 if y > q60, 3 if not visible.

ii.
```python
TONGUE_VISIBLE_THRESHOLD = 0.9
# ...
visible = likelihood >= likelihood_threshold
if np.any(visible):
    q40, q60 = np.quantile(y[visible], [0.4, 0.6]).astype(np.float32)
# ...
frame_idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
# ... classify based on y value at that frame
labels[visible_bins[y_vis < q40]] = 0
mid = (y_vis >= q40) & (y_vis <= q60)
labels[visible_bins[mid]] = 1
labels[visible_bins[y_vis > q60]] = 2
```

iii. The agent uses a likelihood threshold of 0.9 and computes percentiles over raw frame values rather than bin means.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. 0: y < 40th percentile, 1: 40th to 60th percentile (inclusive on both ends), 2: y > 60th percentile, 3: not visible (low likelihood or no frame in bin).

ii.
```python
labels[visible_bins[y_vis < q40]] = 0
mid = (y_vis >= q40) & (y_vis <= q60)
labels[visible_bins[mid]] = 1
labels[visible_bins[y_vis > q60]] = 2
```

iii. The four categories match the instructions' specification.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as neural data. For each trial, bin boundaries are defined as `go_time + BIN_EDGES_S`, and video frames are found via `searchsorted` on the camera timestamps. Each bin uses the last frame before the bin's end time.

ii.
```python
bin_starts = go_time + BIN_EDGES_S[:-1]
bin_ends = go_time + BIN_EDGES_S[1:]
frame_idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
```

iii. Using the same go-cue-relative bin grid as the neural data ensures alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Session with no good units or no annotated units**: Dropped (returns None).
- **Sessions with fewer recorded trials than behavioral trials**: Truncated to recorded count using `is_good_trials`.
- **Trials with all-zero neural data after binning**: Removed post-hoc.
- **Missing photostim fields**: Trials with `N/A` get all-zero photostim input.
- **Missing sample onset**: Falls back to median go-minus-sample offset.
- **Low-likelihood tongue frames**: Assigned "not visible" (class 3).

ii.
```python
# Missing units/annotations
if len(good_unit_indices) == 0:
    return None, brain_region_to_idx

# All-zero neural trials
valid_trial_indices = [i for i, trial in enumerate(neural_trials) if np.any(trial)]

# Missing sample onset fallback
per_trial[i] = go_times[i] - median_go_minus_sample
```

iii. The agent handles missing data case-by-case, preferring to drop or fill with sensible defaults rather than fabricating values.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file (loading spike times and tongue tracking data from HDF5) is the dominant cost. The agent's approach of reading raw HDF5 index arrays rather than `units.to_dataframe()` was an optimization to avoid dragging in ragged spike-time columns.

ii. N/A

iii. The agent explicitly optimized by avoiding `to_dataframe()` on units.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit loop in `build_neural_trials` processes one unit at a time, though within each unit it vectorizes across trials using `searchsorted` and `np.add.at`. The per-trial loop in `build_tongue_binned_labels` processes one trial at a time and the per-trial loop in the main conversion for building input/output arrays.

ii.
```python
for unit_pos, unit_idx in enumerate(good_unit_indices):
    # ... vectorized across trials within
    np.add.at(counts[unit_pos], (trial_idx[good], bin_idx[good]), 1)

for go_time in go_times:
    # per-trial tongue processing
```

iii. The per-unit loop is unavoidable due to ragged spike storage.

## 10-c. What processing does the code repeat multiple times?

i. The trials dataframe is processed multiple times with `.astype(str).to_numpy()` calls for different columns, and some trial-level processing happens in both the filtering stage and the output construction. The `compute_control_performance` function computes trial statistics that are stored in metadata but not used for filtering.

ii.
```python
outcome_strings = trials_df["outcome"].astype(str).to_numpy()
early_strings = trials_df["early_lick"].astype(str).to_numpy()
```

iii. Minor redundancy; not a significant performance concern.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `compute_control_performance` function calculates per-session behavioral performance statistics (hit rates, left/right hit counts) that are stored in metadata but not used by the decoder or downstream analyses.

ii.
```python
def compute_control_performance(trials_df) -> tuple[float, int, int, int]:
    # ... computes performance stats stored in session_info metadata
    return performance, left_hits, right_hits, n_regular
```

iii. These statistics are informational metadata only.
