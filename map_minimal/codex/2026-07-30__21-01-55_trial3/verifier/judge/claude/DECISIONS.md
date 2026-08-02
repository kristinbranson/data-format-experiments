# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads NWB files from the data directory by globbing `sub-*/*.nwb`. Each NWB file is opened with `h5py` and contains units (spike times, classification, annotations), trial intervals (start/stop times, instructions, outcomes, etc.), behavioral events (go cue, sample start, lick times), and video tracking data. All sessions across all subjects are processed in sorted order.

ii.
```python
for path in sorted(data_dir.glob("sub-*/*.nwb")):
    session, stats = build_session(path, trial_limit=trial_limit)
```
Inside `build_session`:
```python
with h5py.File(path, "r") as f:
    units = f["units"]
    trials = f["intervals/trials"]
    go_times = np.asarray(f["acquisition/BehavioralEvents/go_start_times/timestamps"][()], ...)
    # etc.
```

iii. The AI's CONVERSION_NOTES.md states: "Included every NWB session with at least one unit whose `units/classification == 'good'`." The AI loads directly from NWB files (the data format provided on DANDI), rather than from the .mat export format used in the reference code. This is appropriate since the NWB files are the provided data.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the NWB file's parent directory name (e.g., `sub-440956`), stripping the `sub-` prefix. All unique subjects are collected and sorted alphabetically.

ii.
```python
stats["subject"] = path.parent.name.replace("sub-", "")
subjects = sorted({s["subject"] for s in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The subject identity is derived directly from the DANDI directory structure, which organizes files by subject.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Each session is processed independently by `build_session()`. Sessions with zero good units or fewer than 2 valid trials are skipped.

ii.
```python
for path in sorted(data_dir.glob("sub-*/*.nwb")):
    session, stats = build_session(path, trial_limit=trial_limit)
    if session is not None:
        sessions.append(session)
```

iii. CONVERSION_NOTES: "Included every NWB session with at least one unit whose `units/classification == 'good'`. This yielded 173 included sessions and 1 skipped session."

## 1-d. How are the data split into trials?

i. Trials are read from the `intervals/trials` table in each NWB file. The trial count is truncated to match the number of trials covered by the ephys data (`units/is_good_trials.shape[1]`). Each trial has associated behavioral events (go cue, lick times, etc.) indexed by trial number.

ii.
```python
ephys_trial_count = int(units["is_good_trials"].shape[1])
trial_start = np.asarray(trials["start_time"][()], dtype=np.float64)[:ephys_trial_count]
trial_stop = np.asarray(trials["stop_time"][()], dtype=np.float64)[:ephys_trial_count]
```

iii. CONVERSION_NOTES: "Some NWBs contain more rows in the trial table than are covered by the ephys trial matrix... To avoid misalignment, all trial-level behavioral/event arrays were truncated to `ephys_trial_count`."

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes: (1) auto_water and free_water trials, (2) trials whose [-2.5, 1.5) window relative to go cue falls outside video timestamps, (3) trials with all-zero neural activity after binning. It retains early-lick, ignore, and photostimulation trials because they are decoder targets/inputs.

ii.
```python
keep_mask = (auto_water == 0) & (free_water == 0)
# ...
if abs_edges[0] < video_timestamps[0] or abs_edges[-1] > video_timestamps[-1]:
    stats["n_trials_dropped_window"] += 1
    continue
# ...
nonzero_mask = np.asarray([np.any(trial != 0) for trial in neural_trials], dtype=bool)
```

iii. CONVERSION_NOTES: "Reference analysis code defines a 'regular trial' mask that excludes early-lick, auto-water, free-water, no-response, and photostimulation trials... For that reason, the conversion: excludes auto_water and free_water trials; keeps early-lick trials; keeps ignore trials; keeps photostimulation trials."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the spike times of units classified as "good" in the NWB `units` table. Spike times are read from `units/spike_times` (a flat array) with indices from `units/spike_times_index` (ragged array indexing).

ii.
```python
classification = decode_bytes_array(units["classification"][()])
good_unit_idx = np.flatnonzero(classification == "good")
# ...
spike_times_ds = units["spike_times"]
spike_times_index_ds = units["spike_times_index"]
# ...
spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
```

iii. CONVERSION_NOTES: "Used the NWB-provided quality-control label `classification == 'good'`. This matches the papers' and white paper's description that downstream analyses use units labeled good by the QC classifier."

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50-ms non-overlapping bins spanning [-2.5, 1.5) seconds relative to go cue onset (80 bins total). Spike counts are converted to firing rates by dividing by bin width (0.05 s). The result is stored as float16.

ii.
```python
BIN_EDGES = fixed_bin_edges(WINDOW_START, WINDOW_END, BIN_WIDTH)  # 81 edges
# ...
edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
counts = np.diff(edge_idx, axis=1)
rates = (counts.astype(np.float32) / BIN_WIDTH).astype(np.float16)
```

iii. CONVERSION_NOTES: "Neural activity is stored as firing rate in spikes/s: `rate = spike_count_in_bin / 0.05`"

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == "good"` are included. No additional firing-rate or waveform filtering is applied. Sessions with zero good units are excluded. Trials with all-zero neural activity across all units are dropped.

ii.
```python
good_unit_idx = np.flatnonzero(classification == "good")
if good_unit_idx.size == 0:
    return None, stats
# ...
nonzero_mask = np.asarray([np.any(trial != 0) for trial in neural_trials], dtype=bool)
```

iii. CONVERSION_NOTES: "No extra firing-rate or waveform filtering was added on top of the NWB `good` label."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go cue onset. For each trial, absolute bin edges are computed as `go_time + BIN_EDGES`, so the window spans [go_time - 2.5, go_time + 1.5). Spikes are counted within these absolute-time bins using `np.searchsorted`.

ii.
```python
go_abs = go_times[valid_trials]
abs_edge_matrix = go_abs[:, None] + BIN_EDGES[None, :]
# ...
edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
counts = np.diff(edge_idx, axis=1)
```

iii. CONVERSION_NOTES: "Alignment event: go cue onset. Window: [-2.5, 1.5) seconds relative to go cue."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50 ms (0.05 s). Bins are non-overlapping (stride equals width). No rebinning is applied; spike times are binned directly into the final resolution. The reference code uses different bin widths (40ms with 3.4ms stride in `preprocess_all_ephys.py`, or 100ms with 50ms stride in `preprocessing_DJ_2022Aug.py`), but the instructions explicitly specify 50-ms bins.

ii.
```python
BIN_WIDTH = 0.05
BIN_STRIDE = 0.05
```

iii. The instructions specify "Use 50-ms-width bins for computing firing rates." The AI follows this directly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Tone onset is derived from `acquisition/BehavioralEvents/sample_start_times/timestamps` in the NWB file. The last sample_start_time that falls between trial start and go cue is used as the tone onset time.

ii.
```python
sample_starts = np.asarray(f["acquisition/BehavioralEvents/sample_start_times/timestamps"][()], dtype=np.float64)
# ...
tone_time_abs, tone_used_fallback = last_sample_before_go(sample_starts, float(trial_start[trial_idx]), go_abs)
```

iii. CONVERSION_NOTES: "The tone onset was taken as the last `sample_start_times` timestamp between trial start and go cue. This follows the task structure in which multiple sample starts can occur due to replay after early licking."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the tone onset time (absolute) is found. Then for each bin center, the time from tone onset is computed as `bin_center_relative_to_go - tone_onset_relative_to_go`. This produces a continuous time-varying signal where zero corresponds to tone onset.

ii.
```python
tone_on_rel = tone_abs[row_idx] - go_abs[row_idx]
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
```

iii. CONVERSION_NOTES: "For each neural bin, the value is: `bin_center_relative_to_go - tone_onset_relative_to_go`. That makes zero correspond to tone onset."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time_from_tone values are computed at the same bin centers as the neural data (80 values matching the 80 neural time bins), so alignment is inherent.

ii.
```python
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
# BIN_CENTERS are the same centers used for neural binning
input_trials.append(np.vstack([time_from_tone, stim_on]).astype(np.float32, copy=False))
```

iii. Same bin centers ensure alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from three trial-level fields: `photostim_onset`, `photostim_duration`, and `photostim_power` from the NWB `intervals/trials` table.

ii.
```python
stim_onset = parse_optional_float_array(trials["photostim_onset"][()])[:ephys_trial_count]
stim_duration = parse_optional_float_array(trials["photostim_duration"][()])[:ephys_trial_count]
stim_power = parse_optional_float_array(trials["photostim_power"][()])[:ephys_trial_count]
```

iii. CONVERSION_NOTES: "Derived from `photostim_onset`, `photostim_duration`, and `photostim_power`."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. If a trial has finite stim_power > 0 and finite onset/duration, the stimulation interval is computed relative to go cue: `on_rel = stim_onset - go_rel` and `off_rel = on_rel + stim_duration`. A binary vector is created where bin centers within [on_rel, off_rel) are marked as 1. Non-stimulation trials get all zeros.

ii.
```python
if np.isfinite(stim_power[trial_idx]) and stim_power[trial_idx] > 0 and np.isfinite(stim_onset[trial_idx]) and np.isfinite(stim_duration[trial_idx]):
    go_rel = go_abs - float(trial_start[trial_idx])
    on_rel = float(stim_onset[trial_idx] - go_rel)
    off_rel = on_rel + float(stim_duration[trial_idx])
# ...
stim_on = ((BIN_CENTERS >= stim_on_rel[row_idx]) & (BIN_CENTERS < stim_off_rel[row_idx])).astype(np.float32)
```

iii. CONVERSION_NOTES: "Stimulation times are converted from trial-start reference into go-cue reference, matching the reference code pattern."

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The binary photostimulation vector is computed at the same 80 bin centers as the neural data, ensuring alignment.

ii.
```python
stim_on = ((BIN_CENTERS >= stim_on_rel[row_idx]) & (BIN_CENTERS < stim_off_rel[row_idx])).astype(np.float32)
```

iii. Same bin centers as neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `acquisition/BehavioralEvents/left_lick_times/timestamps` and `acquisition/BehavioralEvents/right_lick_times/timestamps`. The first post-go-cue lick within the trial determines the choice. For ignore trials (no post-go lick), the fallback is the instructed side from `intervals/trials/trial_instruction`.

ii.
```python
left_licks = np.asarray(f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()], dtype=np.float64)
right_licks = np.asarray(f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()], dtype=np.float64)
# ...
choice_code, choice_used_fallback = first_post_go_choice(left_licks, right_licks, go_abs, float(trial_stop[trial_idx]), trial_instruction[trial_idx])
```

iii. CONVERSION_NOTES: "Choice is the first post-go lick side within the trial; ignore trials fall back to the instructed side because the target format requires a binary choice label."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The `first_post_go_choice` function finds the first left and right lick after go cue within the trial. If both exist, the earlier one determines choice. If only one exists, that side is chosen. If neither exists (ignore trial), the instructed side is used as fallback. Left = 0, right = 1. The choice is per-trial and expanded to all 80 time bins.

ii.
```python
def first_post_go_choice(left_licks, right_licks, go_time, stop_time, instruction):
    left_idx = np.searchsorted(left_licks, go_time, side="left")
    right_idx = np.searchsorted(right_licks, go_time, side="left")
    left_time = left_licks[left_idx] if left_idx < len(left_licks) and left_licks[left_idx] <= stop_time else np.nan
    right_time = right_licks[right_idx] if right_idx < len(right_licks) and right_licks[right_idx] <= stop_time else np.nan
    if np.isfinite(left_time) and np.isfinite(right_time):
        return (0, False) if left_time <= right_time else (1, False)
    # ... fallback to instruction
    return (0 if instruction == "left" else 1), True
```

iii. The fallback for ignore trials is reasonable given the binary label requirement.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `intervals/trials/outcome` in the NWB file, which contains string values "ignore", "miss", or "hit".

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

iii. Direct mapping from the NWB outcome field.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String outcome values are mapped to integers: ignore=0, miss=1, hit=2. The value is per-trial and expanded to all 80 time bins.

ii.
```python
np.full(BIN_CENTERS.shape[0], outcome_codes[row_idx], dtype=np.int8)
```

iii. Straightforward categorical encoding matching the instructions.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. This question references "Distance to reward zone" but the actual output is "Outcome" which is a per-trial value. It is simply replicated across all 80 time bins to match neural data dimensions.

ii.
```python
np.full(BIN_CENTERS.shape[0], outcome_codes[row_idx], dtype=np.int8)
```

iii. Per-trial value broadcast to time-varying format.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from `intervals/trials/early_lick` in the NWB file, which contains string values like "early" or "no early".

ii.
```python
early_lick = decode_bytes_array(trials["early_lick"][()])[:ephys_trial_count]
# ...
early_code = 1 if early_lick[trial_idx] == "early" else 0
```

iii. Direct from the NWB trial table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string "early" is mapped to 1, anything else to 0. The value is per-trial and expanded to all 80 time bins.

ii.
```python
early_code = 1 if early_lick[trial_idx] == "early" else 0
# ...
np.full(BIN_CENTERS.shape[0], early_codes[row_idx], dtype=np.int8)
```

iii. Binary encoding matching instructions (no=0, yes=1).

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` (column index 1 for y-coordinate) and its associated `timestamps`.

ii.
```python
tongue_data = np.asarray(
    f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"][()],
    dtype=np.float64,
)
tongue_y = tongue_data[:, 1]
video_timestamps = np.asarray(
    f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps"][()],
    dtype=np.float64,
)
```

iii. Uses the side-camera tongue tracking data from the NWB file.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Session-level 40th and 60th percentiles of the full tongue_y array are computed. For each trial, the tongue y value at each time bin is extracted (last video frame within each bin) and discretized into 3 categories.

ii.
```python
tongue_p40, tongue_p60 = np.percentile(tongue_y, [40.0, 60.0])
# ...
tongue_cat = trial_tongue_categories(video_timestamps, tongue_y, abs_edges, tongue_p40, tongue_p60)
```

iii. CONVERSION_NOTES: "Percentiles were computed over the full session's raw `tongue_y` values."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Values below the 40th percentile are category 0, between 40th and 60th percentile are category 1, above 60th percentile are category 2.

ii.
```python
cats = np.zeros(values.shape[0], dtype=np.int8)
cats[values > p60] = 2
mid = (values >= p40) & (values <= p60)
cats[mid] = 1
```

iii. Matches the instructions: 0 = < 40th pct, 1 = 40th-60th pct, 2 = > 60th pct.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each neural bin, the tongue y value is sampled as the last video frame before the bin's right edge. `np.searchsorted` is used to find video frame indices corresponding to bin edges, and the last frame within each bin is used.

ii.
```python
def trial_tongue_categories(video_timestamps, tongue_y, abs_edges, p40, p60):
    starts = np.searchsorted(video_timestamps, abs_edges[:-1], side="left")
    ends = np.searchsorted(video_timestamps, abs_edges[1:], side="left")
    for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
        if end_idx > start_idx:
            values[i] = tongue_y[end_idx - 1]
        else:
            fallback_idx = max(0, min(len(tongue_y) - 1, end_idx - 1))
            values[i] = tongue_y[fallback_idx]
```

iii. CONVERSION_NOTES: "For each neural bin, took the last video frame within that bin."

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several fallback mechanisms handle missing data:
- Tone onset: if no sample_start_times found between trial start and go cue, a fallback of `go_time - 1.85` is used (representing typical sample-to-go delay).
- Choice: if no post-go licks found (ignore trials), the instructed side is used as fallback.
- Stim parameters: if any of power/onset/duration is NaN or power is 0, the trial is treated as non-stimulation.
- Tongue tracking: if no video frame falls within a bin, the nearest frame is used.
- All-zero neural trials are dropped.
- Sessions with <2 valid trials are excluded.

ii.
```python
# Tone fallback
return float(go_time - 1.85), True
# Choice fallback
return (0 if instruction == "left" else 1), True
# Tongue fallback
fallback_idx = max(0, min(len(tongue_y) - 1, end_idx - 1))
```

iii. Fallback counts are tracked in stats for transparency.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is spike binning: for each good unit, the code reads the full spike train and uses `searchsorted` to count spikes in all trial bins. With ~400 units per session and ~500 trials, this involves many searchsorted operations on potentially large spike time arrays.

ii.
```python
for unit_row, unit_idx in enumerate(good_unit_idx):
    spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
    edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
```

iii. The AI processes ~69,000 units across 173 sessions, each requiring I/O from HDF5 and searchsorted operations.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop over units (`for unit_row, unit_idx in enumerate(good_unit_idx)`) could potentially be parallelized but not easily vectorized since each unit has a different-length spike train stored in a ragged array. The inner loop copying rates into per-trial matrices (`for trial_row in range(n_trials)`) could be replaced with array slicing.

ii.
```python
for unit_row, unit_idx in enumerate(good_unit_idx):
    # ... per unit processing
    for trial_row in range(n_trials):
        neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

iii. The inner trial loop is a straightforward assignment that could use list comprehension or direct array assignment, but the performance impact is minor compared to I/O.

## 10-c. What processing does the code repeat multiple times?

i. The code reads the full session's video timestamps and tongue data once per session, which is efficient. However, `np.searchsorted` on `sample_starts` and lick times is called per trial in a loop, when it could be vectorized across trials. The `ragged_row` function is called per unit, reading from HDF5 each time.

ii.
```python
for trial_idx in keep_idx:
    # Per-trial: searchsorted on sample_starts, left_licks, right_licks
    tone_time_abs, tone_used_fallback = last_sample_before_go(sample_starts, ...)
    choice_code, choice_used_fallback = first_post_go_choice(left_licks, right_licks, ...)
```

iii. The repeated searchsorted calls on the same sorted arrays with different query points could be batched.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores neural data as float16 firing rates, which may lose precision unnecessarily. The `BIN_STRIDE` constant is defined but never used (stride equals width). The code computes `session_output_counts` (Counter) that is never included in the output. The `trial_indices_source` and `tongue_percentiles` are stored per session but not included in the final pickle output.

ii.
```python
BIN_STRIDE = 0.05  # defined but unused
session_output_counts = Counter()  # populated but not returned
session = {
    ...
    "trial_indices_source": valid_trials,  # not in final output
    "tongue_percentiles": (float(tongue_p40), float(tongue_p60)),  # not in final output
}
```

iii. These are minor inefficiencies; the intermediate session data is discarded when assembling the final dictionary.
