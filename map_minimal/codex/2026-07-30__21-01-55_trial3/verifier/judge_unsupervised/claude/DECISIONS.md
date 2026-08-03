# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB (HDF5) files stored in `/app/data/`. It iterates over all files matching the glob pattern `sub-*/*.nwb` under the data directory. Each NWB file represents one session. The files are opened with `h5py.File` and fields are read directly from the HDF5 structure (units, intervals/trials, acquisition/BehavioralEvents, acquisition/BehavioralTimeSeries). This differs from the reference code which loads from MATLAB `.mat` files exported from DataJoint, but the NWB files contain equivalent information.

ii.
```python
def convert_dataset(data_dir: Path, session_limit: int | None = None, trial_limit: int | None = None):
    sessions = []
    session_stats = []
    for path in sorted(data_dir.glob("sub-*/*.nwb")):
        session, stats = build_session(path, trial_limit=trial_limit)
        session_stats.append(stats)
        if session is not None:
            sessions.append(session)
```

iii. The agent noted that the NWB files are the published data from the MAP dataset (DANDI archive). The agent reads directly from HDF5 using h5py rather than pynwb for performance reasons, after confirming the structure matches.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the parent directory name of each NWB file (e.g., `sub-440956`). The `sub-` prefix is stripped to get the subject ID. All unique subjects are collected, sorted, and indexed.

ii.
```python
stats["subject"] = path.parent.name.replace("sub-", "")
# ...
subjects = sorted({s["subject"] for s in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The agent used the directory structure of the NWB dataset, which organizes files by subject. This matches the BIDS-like organization of the DANDI archive data.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions with zero "good" units (based on `units/classification == "good"`) or fewer than 2 valid trials after filtering are excluded. This resulted in 173 included sessions and 1 skipped session (which had zero good units).

ii.
```python
good_unit_idx = np.flatnonzero(classification == "good")
stats["n_units_good"] = int(good_unit_idx.size)
if good_unit_idx.size == 0:
    return None, stats
# ...
if len(valid_trials) < 2:
    return None, stats
```

iii. The agent noted that the paper reports 173 behavioral sessions and matched this count. The agent did NOT apply the paper's strict session-selection criteria (>65% performance and >=50 correct left/right trials) because it found only 145 of 174 NWB files pass that rule, and the NWB dataset already represents the published sessions.

## 1-d. How are the data split into trials?

i. Trial information is read from `intervals/trials` in each NWB file. Trial arrays (start_time, stop_time, trial_instruction, early_lick, outcome, auto_water, free_water, photostim fields) are truncated to match the ephys trial count from `units/is_good_trials.shape[1]` to avoid misalignment between behavioral and neural data.

ii.
```python
ephys_trial_count = int(units["is_good_trials"].shape[1])
trials = f["intervals/trials"]
trial_start = np.asarray(trials["start_time"][()], dtype=np.float64)[:ephys_trial_count]
trial_stop = np.asarray(trials["stop_time"][()], dtype=np.float64)[:ephys_trial_count]
# ... (similarly for other trial fields)
```

iii. The agent discovered that some NWB files contain more trial rows in the behavioral table than are covered by the ephys recording. Truncating to `ephys_trial_count` prevents misalignment where late behavioral trials would have no corresponding neural data.

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes: (1) auto_water and free_water trials, (2) trials whose aligned window [-2.5, 1.5) falls outside side-camera timestamps, (3) trials with all-zero neural activity after binning. It retains early-lick, ignore (no-response), and photostimulation trials because these are decoder targets/inputs.

ii.
```python
keep_mask = (auto_water == 0) & (free_water == 0)
# ...
if abs_edges[0] < video_timestamps[0] or abs_edges[-1] > video_timestamps[-1]:
    stats["n_trials_dropped_window"] += 1
    continue
# ... (later)
nonzero_mask = np.asarray([np.any(trial != 0) for trial in neural_trials], dtype=bool)
```

iii. The agent explicitly reasoned that the reference code's `get_regular_trial_mask` excludes early_lick, no-response, and stimulation trials, but the decoder task instructions require early_lick as an output, outcome including ignore, and photostim as an input. Therefore, the agent correctly deviated from the reference trial mask to keep these trial types.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index` in the NWB files. Only units with `units/classification == "good"` are included.

ii.
```python
classification = decode_bytes_array(units["classification"][()])
good_unit_idx = np.flatnonzero(classification == "good")
# ...
spike_times_ds = units["spike_times"]
spike_times_index_ds = units["spike_times_index"]
```

iii. The agent used the NWB-provided quality-control label matching the paper's description that downstream analyses use units labeled "good" by the QC classifier.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50-ms non-overlapping bins spanning [-2.5, 1.5) seconds relative to the go cue. Spike counts in each bin are converted to firing rates by dividing by bin width (0.05 s). The result is stored as float16.

ii.
```python
BIN_EDGES = fixed_bin_edges(WINDOW_START, WINDOW_END, BIN_WIDTH)  # 81 edges for 80 bins
# ...
edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
counts = np.diff(edge_idx, axis=1)
rates = (counts.astype(np.float32) / BIN_WIDTH).astype(np.float16)
```

iii. The agent's approach uses exact bin edges [edge_i, edge_{i+1}) and counts spikes via searchsorted, which is equivalent to the reference code's `sliding_histogram` with stride=bin_width. The reference code defines bins by [center-width/2, center+width/2) and counts with `np.sum(np.all([trial>=binInt[0], trial<binInt[1]], axis=0))`, which is the same half-open interval convention.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == "good"` are included. No additional firing-rate or waveform filtering is applied. Trials that produce all-zero neural activity across all units are dropped.

ii.
```python
good_unit_idx = np.flatnonzero(classification == "good")
# ...
nonzero_mask = np.asarray([np.any(trial != 0) for trial in neural_trials], dtype=bool)
if not np.all(nonzero_mask):
    neural_trials = [trial for trial, keep in zip(neural_trials, nonzero_mask) if keep]
```

iii. The agent relied on the NWB `good` label as the sole unit QC filter, consistent with the paper's classifier-based quality control. The all-zero filter is an additional safeguard against degenerate trials.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. The go cue times are read from `acquisition/BehavioralEvents/go_start_times/timestamps`. Absolute bin edges are computed as go_time + BIN_EDGES, where BIN_EDGES spans [-2.5, 1.5).

ii.
```python
go_times = np.asarray(f["acquisition/BehavioralEvents/go_start_times/timestamps"][()], dtype=np.float64)[:ephys_trial_count]
# ...
abs_edge_matrix = go_abs[:, None] + BIN_EDGES[None, :]
```

iii. The agent correctly identified the go cue as the alignment event per the instructions. The reference code also uses go cue alignment (spike times in the raw data are described as "relative to go cue time").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 50 ms (0.05 s), yielding 80 bins for the [-2.5, 1.5) window. No overlapping bins or stride different from bin width is used (stride = bin_width = 0.05 s). No rebinning is applied.

ii.
```python
BIN_WIDTH = 0.05
BIN_STRIDE = 0.05
# ...
n_bins = int(round((end_time - start_time) / width))  # = 80
```

iii. The 50-ms bin size matches the instruction requirement. The reference code supports configurable bin_width and stride; the agent used non-overlapping bins as specified.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `acquisition/BehavioralEvents/sample_start_times/timestamps` (tone/sample onset times), `acquisition/BehavioralEvents/go_start_times/timestamps` (go cue times), and `intervals/trials/start_time` (trial start times).

ii.
```python
sample_starts = np.asarray(f["acquisition/BehavioralEvents/sample_start_times/timestamps"][()], dtype=np.float64)
# ...
tone_time_abs, tone_used_fallback = last_sample_before_go(sample_starts, float(trial_start[trial_idx]), go_abs)
```

iii. The agent identified `sample_start_times` as the tone onset events, which matches the task description where the sample epoch contains the instruction tones.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the tone onset is identified as the last `sample_start_times` timestamp between trial start and go cue (to handle replays due to early licking). The relative tone onset time is computed as `tone_abs - go_abs`. Then for each time bin, the value is `bin_center - tone_onset_relative_to_go`, making zero correspond to tone onset.

ii.
```python
def last_sample_before_go(sample_starts, trial_start, go_time):
    lo = np.searchsorted(sample_starts, trial_start, side="left")
    hi = np.searchsorted(sample_starts, go_time, side="right")
    if hi > lo:
        return float(sample_starts[hi - 1]), False
    return float(go_time - 1.85), True

# ...
tone_on_rel = tone_abs[row_idx] - go_abs[row_idx]
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
```

iii. The agent reasoned that multiple sample starts can occur due to early-lick-triggered replays, so taking the last one before the go cue gives the final effective tone onset. The fallback of go_time - 1.85 is based on the typical sample-to-go delay (sample epoch ~0.65s + delay ~1.2s = ~1.85s).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time_from_tone values are computed at each bin center of the neural data, so they are automatically aligned. Each bin center's value = bin_center_relative_to_go - tone_onset_relative_to_go.

ii.
```python
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
input_trials.append(np.vstack([time_from_tone, stim_on]).astype(np.float32, copy=False))
```

iii. The alignment is implicit through using the same BIN_CENTERS for both neural binning and input computation.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `intervals/trials/photostim_onset`, `intervals/trials/photostim_duration`, and `intervals/trials/photostim_power`.

ii.
```python
stim_onset = parse_optional_float_array(trials["photostim_onset"][()])[:ephys_trial_count]
stim_duration = parse_optional_float_array(trials["photostim_duration"][()])[:ephys_trial_count]
stim_power = parse_optional_float_array(trials["photostim_power"][()])[:ephys_trial_count]
```

iii. The agent checked for finite values and positive power to determine whether photostimulation occurred on each trial.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Stimulation onset/duration are converted from trial-start-relative to go-cue-relative timing. A binary indicator is created: 1 if the bin center falls within the stimulation interval [stim_on, stim_off), 0 otherwise. Non-stimulation trials get all zeros.

ii.
```python
go_rel = go_abs - float(trial_start[trial_idx])
on_rel = float(stim_onset[trial_idx] - go_rel)
off_rel = on_rel + float(stim_duration[trial_idx])
# ...
stim_on = ((BIN_CENTERS >= stim_on_rel[row_idx]) & (BIN_CENTERS < stim_off_rel[row_idx])).astype(np.float32)
```

iii. The agent noted that photostim_onset in the NWB is relative to trial start, so it must be converted to go-cue-relative coordinates. This matches the reference code pattern where stimulation times are realigned: `stimulation[i_trial, 2:] -= gocue_time_trial`.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The binary photostimulation indicator is computed at each bin center, matching the neural data's temporal structure.

ii.
```python
stim_on = ((BIN_CENTERS >= stim_on_rel[row_idx]) & (BIN_CENTERS < stim_off_rel[row_idx])).astype(np.float32)
```

iii. Same alignment via bin centers as the tone onset input.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `acquisition/BehavioralEvents/left_lick_times/timestamps` and `acquisition/BehavioralEvents/right_lick_times/timestamps`, with fallback to `intervals/trials/trial_instruction` for ignore trials.

ii.
```python
left_licks = np.asarray(f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()], dtype=np.float64)
right_licks = np.asarray(f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()], dtype=np.float64)
# ...
choice_code, choice_used_fallback = first_post_go_choice(left_licks, right_licks, go_abs, float(trial_stop[trial_idx]), trial_instruction[trial_idx])
```

iii. The agent reasoned that choice should be determined from the actual first post-go lick, with a fallback to the instructed side for ignore trials that have no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For each trial, the first lick after the go cue on each side is found. The earlier lick determines the choice (left=0, right=1). If no lick occurs (ignore trial), the instructed side from `trial_instruction` is used as fallback. The choice is per-trial but expanded across all time bins.

ii.
```python
def first_post_go_choice(left_licks, right_licks, go_time, stop_time, instruction):
    left_idx = np.searchsorted(left_licks, go_time, side="left")
    right_idx = np.searchsorted(right_licks, go_time, side="left")
    left_time = left_licks[left_idx] if left_idx < len(left_licks) and left_licks[left_idx] <= stop_time else np.nan
    right_time = right_licks[right_idx] if right_idx < len(right_licks) and right_licks[right_idx] <= stop_time else np.nan
    if np.isfinite(left_time) and np.isfinite(right_time):
        return (0, False) if left_time <= right_time else (1, False)
    if np.isfinite(left_time):
        return 0, False
    if np.isfinite(right_time):
        return 1, False
    return (0 if instruction == "left" else 1), True
```

iii. The agent needed a fallback for ignore trials because the output format requires a binary choice label. Using the instructed side is a reasonable fallback.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `intervals/trials/outcome` field in the NWB files.

ii.
```python
outcome = decode_bytes_array(trials["outcome"][()])[:ephys_trial_count]
# ...
if outcome[trial_idx] == "ignore":
    outcome_code = 0
elif outcome[trial_idx] == "miss":
    outcome_code = 1
elif outcome[trial_idx] == "hit":
    outcome_code = 2
```

iii. The agent read the outcome strings directly from the NWB trial table and mapped them to the integer codes specified in the instructions.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct string-to-integer mapping: ignore=0, miss=1, hit=2. The value is per-trial and expanded across all time bins.

ii.
```python
np.full(BIN_CENTERS.shape[0], outcome_codes[row_idx], dtype=np.int8)
```

iii. Straightforward mapping matching the instruction specification.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from `intervals/trials/early_lick` field in the NWB files.

ii.
```python
early_lick = decode_bytes_array(trials["early_lick"][()])[:ephys_trial_count]
# ...
early_code = 1 if early_lick[trial_idx] == "early" else 0
```

iii. The agent reads the early_lick string from the trial table and converts it to binary.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary mapping: "early" -> 1, anything else -> 0. Per-trial, expanded across all time bins.

ii.
```python
early_code = 1 if early_lick[trial_idx] == "early" else 0
# ...
np.full(BIN_CENTERS.shape[0], early_codes[row_idx], dtype=np.int8)
```

iii. Simple binary classification matching the instruction's no=0, yes=1 scheme.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` (column index 1 for y-coordinate) and `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps`.

ii.
```python
video_timestamps = np.asarray(
    f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps"][()],
    dtype=np.float64,
)
tongue_data = np.asarray(
    f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"][()],
    dtype=np.float64,
)
tongue_y = tongue_data[:, 1]
```

iii. The agent identified the side-camera tongue tracking data from the NWB file. The y-coordinate (column 1) is used as specified.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each neural bin, the last video frame within that bin is used to get the tongue y-value. If a bin has no new frame, a fallback to the nearest available frame is used. Session-wide 40th and 60th percentiles are computed over the full raw tongue_y array for discretization.

ii.
```python
tongue_p40, tongue_p60 = np.percentile(tongue_y, [40.0, 60.0])
# ...
def trial_tongue_categories(video_timestamps, tongue_y, abs_edges, p40, p60):
    starts = np.searchsorted(video_timestamps, abs_edges[:-1], side="left")
    ends = np.searchsorted(video_timestamps, abs_edges[1:], side="left")
    values = np.empty(len(starts), dtype=np.float64)
    for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
        if end_idx > start_idx:
            values[i] = tongue_y[end_idx - 1]
        else:
            fallback_idx = max(0, min(len(tongue_y) - 1, end_idx - 1))
            values[i] = tongue_y[fallback_idx]
```

iii. The agent used the last frame in each bin as the representative tongue position, then discretized using session-wide percentiles.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session discretization using the 40th and 60th percentiles of the full session's raw tongue_y values:
- 0: < 40th percentile
- 1: between 40th and 60th percentile (inclusive)
- 2: > 60th percentile

ii.
```python
tongue_p40, tongue_p60 = np.percentile(tongue_y, [40.0, 60.0])
# ...
cats = np.zeros(values.shape[0], dtype=np.int8)
cats[values > p60] = 2
mid = (values >= p40) & (values <= p60)
cats[mid] = 1
```

iii. The thresholds match the instruction specification: < 40th → 0, 40th-60th → 1, > 60th → 2.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The tongue y-position is sampled at each neural bin edge interval and discretized, producing one category value per neural bin. The absolute bin edges are computed from go_cue + BIN_EDGES, matching the neural data alignment.

ii.
```python
abs_edge_matrix = go_abs[:, None] + BIN_EDGES[None, :]
# ...
tongue_cat = trial_tongue_categories(
    video_timestamps=video_timestamps,
    tongue_y=tongue_y,
    abs_edges=abs_edge_matrix[row_idx],
    p40=tongue_p40,
    p60=tongue_p60,
)
```

iii. Alignment is achieved through the same absolute bin edges used for neural data binning.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several fallback mechanisms:
- Missing tone onset (no sample_start_times between trial start and go cue): fallback to go_time - 1.85s (typical sample-to-go delay)
- Missing choice for ignore trials: fallback to instructed lick direction
- Missing photostim fields (NaN values): treated as non-stimulation trial (all zeros)
- Trials outside video timestamp range: dropped
- All-zero neural trials: dropped
- NWB trials exceeding ephys coverage: truncated to ephys trial count
- `parse_optional_float_array` handles "N/A", "nan", "NaN", empty strings as NaN

ii.
```python
def parse_optional_float_array(arr) -> np.ndarray:
    # ...
    if value in {"N/A", "nan", "NaN", ""}:
        out[i] = np.nan
    else:
        out[i] = float(value)
# ...
return float(go_time - 1.85), True  # tone fallback
return (0 if instruction == "left" else 1), True  # choice fallback
```

iii. The agent documented all fallback cases and tracked their frequency in statistics (e.g., 0 tone fallbacks, 13,064 choice fallbacks on ignore trials in the full dataset).

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the spike binning loop, which iterates over every good unit in every session, loads its spike times, and computes bin counts via searchsorted for all trials. This is O(n_units * n_trials) per session with large spike time arrays. The HDF5 I/O for loading spike times (ragged arrays) is also significant.

ii.
```python
for unit_row, unit_idx in enumerate(good_unit_idx):
    spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
    edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
    counts = np.diff(edge_idx, axis=1)
    rates = (counts.astype(np.float32) / BIN_WIDTH).astype(np.float16)
    for trial_row in range(n_trials):
        neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

iii. The agent vectorized the binning within each unit (all trials at once via flat_abs_edges), but the outer loop over units is inherent due to the ragged spike time storage.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops could potentially be improved:
1. The inner loop `for trial_row in range(n_trials)` that assigns rates per trial could be replaced with array assignment.
2. The `trial_tongue_categories` function loops over bins with `for i, (start_idx, end_idx) in enumerate(zip(starts, ends))` which could potentially be vectorized.
3. The `decode_bytes_array` and `parse_optional_float_array` functions use Python loops over arrays.

ii.
```python
for trial_row in range(n_trials):
    neural_trials[trial_row][unit_row, :] = rates[trial_row]
# Could be replaced with direct indexing if neural_trials were a 3D array

for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
    if end_idx > start_idx:
        values[i] = tongue_y[end_idx - 1]
```

iii. The agent chose list-of-arrays representation for neural_trials (matching the target format), which prevents full vectorization of the assignment loop.

## 10-c. What processing does the code repeat multiple times?

i. The code does not significantly repeat processing. Each session is processed once. However, within `build_session`, the `abs_edge_matrix` is computed for all valid trials and reused. The `BIN_EDGES` and `BIN_CENTERS` are computed once as module-level constants.

ii.
```python
BIN_EDGES = fixed_bin_edges(WINDOW_START, WINDOW_END, BIN_WIDTH)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2.0
```

iii. The code is reasonably efficient in avoiding redundant computation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
1. The all-zero neural trial check loads and processes neural data for trials that are then discarded.
2. The `trial_tongue_categories` function is called for all valid trials including ones that may be dropped by the all-zero filter.
3. The `first_post_go_choice` function computes choice from actual lick times even for trials where the outcome is "ignore" — but this is necessary for correctness since some ignore trials do have post-go licks.
4. The video timestamp boundary check (`abs_edges[0] < video_timestamps[0]`) may unnecessarily exclude some trials if only a small portion of the window falls outside video coverage.
5. The `summarize_conversion` function iterates over all output data to compute histograms, which is diagnostic only.

ii.
```python
# All-zero check happens AFTER all processing
nonzero_mask = np.asarray([np.any(trial != 0) for trial in neural_trials], dtype=bool)
```

iii. These are minor inefficiencies. The main pipeline is straightforward with minimal wasted computation.
