# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over all `sub-*/*.nwb` files under the data directory, sorted lexicographically, and opens each file directly with `h5py`. Within each file it reads the `units` group, the trial table under `intervals/trials`, the go/sample/lick behavioral event timestamps, and the side-camera tongue tracking timestamps/data. It does not use `pynwb`; it accesses the HDF5 paths directly.

ii.
```python
for path in sorted(data_dir.glob("sub-*/*.nwb")):
    session, stats = build_session(path, trial_limit=trial_limit)
```

```python
with h5py.File(path, "r") as f:
    units = f["units"]
    ...
    trials = f["intervals/trials"]
    ...
    go_times = np.asarray(f["acquisition/BehavioralEvents/go_start_times/timestamps"][()], dtype=np.float64)[:ephys_trial_count]
    sample_starts = np.asarray(f["acquisition/BehavioralEvents/sample_start_times/timestamps"][()], dtype=np.float64)
    left_licks = np.asarray(f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()], dtype=np.float64)
    right_licks = np.asarray(f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()], dtype=np.float64)
    video_timestamps = np.asarray(
        f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps"][()],
        dtype=np.float64,
    )
    tongue_data = np.asarray(
        f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"][()],
        dtype=np.float64,
    )
```

iii. `CONVERSION_NOTES.md` says the goal was to convert NWB sessions while matching the released data and paper/code logic as closely as possible. The trajectory shows the AI intentionally inspected the NWB schema directly and decided the files already carried the needed trial table and QC metadata, so direct HDF5 reads were sufficient.

## 1-b. How are the data split into subjects?

i. The AI treats the directory name `sub-<id>` as the subject identifier for each session. It stores that string per session, then builds `subjects` as the sorted unique set and `subject_idx` as the per-session index into that set.

ii.
```python
stats = {
    "session_id": path.stem,
    "subject": path.parent.name.replace("sub-", ""),
    ...
}
```

```python
subjects = sorted({s["subject"] for s in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.asarray([subject_to_idx[session["subject"]] for session in sessions], dtype=np.int16),
```

iii. `CONVERSION_NOTES.md` explicitly says subjects come from the NWB subject directory names. There is no extra grouping logic beyond the one-file-per-session layout.

## 1-c. How are the data split into sessions?

i. The AI uses one NWB file as one session. Session order follows the sorted file list, and the per-session identifier it carries forward is `path.stem` rather than `nwb.identifier`.

ii.
```python
for path in sorted(data_dir.glob("sub-*/*.nwb")):
    session, stats = build_session(path, trial_limit=trial_limit)
```

```python
stats = {
    "session_id": path.stem,
    ...
}
...
session = {
    "session_id": path.stem,
    ...
}
```

iii. `CONVERSION_NOTES.md` describes conversion at the NWB-session level and reports a count of included versus skipped sessions, which shows the AI treated file boundaries as session boundaries.

## 1-d. How are the data split into trials?

i. The AI does not use the full trial table directly. Instead, it truncates every trial-level array to `units["is_good_trials"].shape[1]`, which it interprets as the count of trials covered by ephys. It then loops over the surviving trial indices and builds per-trial inputs, outputs, and neural matrices from those truncated arrays.

ii.
```python
ephys_trial_count = int(units["is_good_trials"].shape[1])

trial_start = np.asarray(trials["start_time"][()], dtype=np.float64)[:ephys_trial_count]
trial_stop = np.asarray(trials["stop_time"][()], dtype=np.float64)[:ephys_trial_count]
trial_instruction = decode_bytes_array(trials["trial_instruction"][()])[:ephys_trial_count]
early_lick = decode_bytes_array(trials["early_lick"][()])[:ephys_trial_count]
outcome = decode_bytes_array(trials["outcome"][()])[:ephys_trial_count]
...
go_times = np.asarray(f["acquisition/BehavioralEvents/go_start_times/timestamps"][()], dtype=np.float64)[:ephys_trial_count]
```

```python
keep_idx = np.flatnonzero(keep_mask)
...
for trial_idx in keep_idx:
    go_abs = float(go_times[trial_idx])
    ...
    valid_trials.append(trial_idx)
```

iii. `CONVERSION_NOTES.md` calls this a “critical correction,” arguing that some NWBs have more trial-table rows than are covered by the ephys trial matrix and that truncating to `is_good_trials.shape[1]` avoids misalignment. The trajectory also shows the AI explicitly decided that `is_good_trials` gave the “actual ephys-covered trial count.”

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes `auto_water` and `free_water` trials up front, keeps early-lick, ignore, and photostimulation trials, then applies two additional mechanical filters: drop trials whose `[-2.5, 1.5)` go-cue-aligned window extends outside the side-camera timestamps, and drop trials whose binned neural activity is all zeros. Sessions with fewer than two remaining trials are dropped.

ii.
```python
keep_mask = (auto_water == 0) & (free_water == 0)
stats["n_trials_dropped_auto_free"] = int(np.sum(~keep_mask))
keep_idx = np.flatnonzero(keep_mask)
```

```python
if abs_edges[0] < video_timestamps[0] or abs_edges[-1] > video_timestamps[-1]:
    stats["n_trials_dropped_window"] += 1
    continue
```

```python
nonzero_mask = np.asarray([np.any(trial != 0) for trial in neural_trials], dtype=bool)
if not np.all(nonzero_mask):
    stats["n_trials_dropped_all_zero_neural"] = int(np.sum(~nonzero_mask))
    neural_trials = [trial for trial, keep in zip(neural_trials, nonzero_mask) if keep]
    input_trials = [trial for trial, keep in zip(input_trials, nonzero_mask) if keep]
    output_trials = [trial for trial, keep in zip(output_trials, nonzero_mask) if keep]
```

```python
if len(valid_trials) < 2:
    return None, stats
...
if n_trials < 2:
    return None, stats
```

iii. `CONVERSION_NOTES.md` explicitly justifies keeping early-lick, ignore, and photostim trials because those variables are required for the decoder, while excluding `auto_water` and `free_water`. The notes also justify the additional video-window and all-zero-neural exclusions as mechanical safeguards.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `units["spike_times"]` plus `units["spike_times_index"]`, using go-cue timestamps to define the trial-aligned bin edges. It only uses units with `classification == "good"`.

ii.
```python
classification = decode_bytes_array(units["classification"][()])
good_unit_idx = np.flatnonzero(classification == "good")
```

```python
spike_times_ds = units["spike_times"]
spike_times_index_ds = units["spike_times_index"]
flat_abs_edges = abs_edge_matrix.reshape(-1)
...
spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
```

iii. `CONVERSION_NOTES.md` says the converter uses the NWB `good` QC label directly and stores neural activity as firing rate from 50 ms bins. The trajectory shows the AI explicitly chose `classification == "good"` as the unit filter.

## 2-b. How is the `neural` data processed?

i. For each retained unit, the AI reads its ragged spike train, bins spikes into 50 ms bins using `np.searchsorted` on flattened absolute bin edges for all trials, differences the cumulative counts to get per-bin spike counts, divides by bin width to get spikes/s, and stores the result as `float16`.

ii.
```python
def ragged_row(data_ds, index_ds, row_idx: int) -> np.ndarray:
    stop = int(index_ds[row_idx])
    start = 0 if row_idx == 0 else int(index_ds[row_idx - 1])
    return np.asarray(data_ds[start:stop], dtype=np.float64)
```

```python
for unit_row, unit_idx in enumerate(good_unit_idx):
    spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
    edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
    counts = np.diff(edge_idx, axis=1)
    rates = (counts.astype(np.float32) / BIN_WIDTH).astype(np.float16)
    for trial_row in range(n_trials):
        neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

iii. `CONVERSION_NOTES.md` states that neural activity is stored as firing rate in spikes/s and that the bins are exact-width 50 ms bins over `[start, end)`. No smoothing, normalization, or baseline subtraction is described in either the notes or the code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units whose `units/classification` equals `"good"`. If a session has zero such units, the entire session is skipped.

ii.
```python
classification = decode_bytes_array(units["classification"][()])
good_unit_idx = np.flatnonzero(classification == "good")
stats["n_units_good"] = int(good_unit_idx.size)
if good_unit_idx.size == 0:
    return None, stats
```

iii. `CONVERSION_NOTES.md` explicitly says the converter used the NWB-provided QC label `classification == "good"` and added no extra firing-rate or waveform thresholds. The trajectory shows the AI checked this against the QC paper and session/unit counts.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go cue onset. It forms absolute bin edges by adding the shared relative bin grid `BIN_EDGES` to each trial’s go-cue timestamp, then bins spike times against those absolute edges.

ii.
```python
BIN_EDGES = fixed_bin_edges(WINDOW_START, WINDOW_END, BIN_WIDTH)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2.0
```

```python
go_abs = go_times[valid_trials]
abs_edge_matrix = go_abs[:, None] + BIN_EDGES[None, :]
flat_abs_edges = abs_edge_matrix.reshape(-1)
```

iii. `CONVERSION_NOTES.md` lists the alignment event as go cue onset. The trajectory also says the converter was intentionally built around go-cue alignment and the shared 50 ms grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms non-overlapping bins from `-2.5` s to `+1.5` s relative to go cue, for 80 time bins per trial. No secondary temporal rebinning is applied.

ii.
```python
WINDOW_START = -2.5
WINDOW_END = 1.5
BIN_WIDTH = 0.05
BIN_STRIDE = 0.05
...
BIN_EDGES = fixed_bin_edges(WINDOW_START, WINDOW_END, BIN_WIDTH)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2.0
```

iii. `CONVERSION_NOTES.md` states the alignment window, bin width, and number of bins explicitly and says the converter uses exact-width bins over `[start, end)`.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `sample_start_times`, `start_time`, and go cue timestamps. For each kept trial it looks for the last sample-start timestamp between trial start and the go cue; if none is found it falls back to `go_time - 1.85`.

ii.
```python
def last_sample_before_go(sample_starts: np.ndarray, trial_start: float, go_time: float):
    lo = np.searchsorted(sample_starts, trial_start, side="left")
    hi = np.searchsorted(sample_starts, go_time, side="right")
    if hi > lo:
        return float(sample_starts[hi - 1]), False
    return float(go_time - 1.85), True
```

```python
sample_starts = np.asarray(f["acquisition/BehavioralEvents/sample_start_times/timestamps"][()], dtype=np.float64)
...
tone_time_abs, tone_used_fallback = last_sample_before_go(sample_starts, float(trial_start[trial_idx]), go_abs)
```

iii. `CONVERSION_NOTES.md` justifies using the last `sample_start_times` event between trial start and go cue because early licking can replay the sample epoch. The notes also report that the fallback was never needed on the full dataset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After selecting one tone time per trial, the AI computes the time from tone onset at each neural bin center by subtracting the trial’s tone-to-go offset from `BIN_CENTERS`. This yields a continuous time-varying signal with zero at tone onset.

ii.
```python
tone_on_rel = tone_abs[row_idx] - go_abs[row_idx]
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
...
input_trials.append(np.vstack([time_from_tone, stim_on]).astype(np.float32, copy=False))
```

iii. `CONVERSION_NOTES.md` states the input is continuous, time-varying, and computed as `bin_center_relative_to_go - tone_onset_relative_to_go`, so that zero corresponds to tone onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The signal is defined on the same 80 go-cue-aligned bin centers used for the neural data, so each time point in `time_from_tone` corresponds directly to a neural time bin.

ii.
```python
BIN_EDGES = fixed_bin_edges(WINDOW_START, WINDOW_END, BIN_WIDTH)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2.0
```

```python
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
...
flat_abs_edges = abs_edge_matrix.reshape(-1)
edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
```

iii. The notes describe all inputs and neural data as being built on the same go-cue-relative 50 ms grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from trial-table fields `photostim_onset`, `photostim_duration`, and `photostim_power`, together with `start_time` and go cue timestamps. It only treats a trial as stimulated if onset, duration, and power are finite and `photostim_power > 0`.

ii.
```python
stim_onset = parse_optional_float_array(trials["photostim_onset"][()])[:ephys_trial_count]
stim_duration = parse_optional_float_array(trials["photostim_duration"][()])[:ephys_trial_count]
stim_power = parse_optional_float_array(trials["photostim_power"][()])[:ephys_trial_count]
```

```python
if np.isfinite(stim_power[trial_idx]) and stim_power[trial_idx] > 0 and np.isfinite(stim_onset[trial_idx]) and np.isfinite(stim_duration[trial_idx]):
    go_rel = go_abs - float(trial_start[trial_idx])
    on_rel = float(stim_onset[trial_idx] - go_rel)
    off_rel = on_rel + float(stim_duration[trial_idx])
    stats["stim_trials_kept"] += 1
else:
    on_rel = np.nan
    off_rel = np.nan
```

iii. `CONVERSION_NOTES.md` explicitly states that `photostim_on` is derived from onset, duration, and power, and says the timing is converted from trial-start reference into go-cue reference.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts the stimulation interval into a binary time series on the neural bin centers. A bin is 1 when its center lies within `[stim_on, stim_off)` and 0 otherwise; non-stimulated trials are all zeros.

ii.
```python
if np.isfinite(stim_on_rel[row_idx]):
    stim_on = ((BIN_CENTERS >= stim_on_rel[row_idx]) & (BIN_CENTERS < stim_off_rel[row_idx])).astype(np.float32)
else:
    stim_on = np.zeros(BIN_CENTERS.shape[0], dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` says the input is binary and time-varying and that bins are marked 1 when the bin center lies within the stimulation interval.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI converts the photostimulation onset from trial-start-relative time into go-cue-relative time, then compares those go-cue-relative bounds to the same `BIN_CENTERS` used for the neural data.

ii.
```python
go_rel = go_abs - float(trial_start[trial_idx])
on_rel = float(stim_onset[trial_idx] - go_rel)
off_rel = on_rel + float(stim_duration[trial_idx])
```

```python
stim_on = ((BIN_CENTERS >= stim_on_rel[row_idx]) & (BIN_CENTERS < stim_off_rel[row_idx])).astype(np.float32)
```

iii. The notes state that stimulation times are converted from trial-start reference into go-cue reference to match the neural alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI does not derive choice from the trial table’s `outcome`. Instead, it looks at the first post-go left or right lick event within the trial’s stop time using `left_lick_times` and `right_lick_times`. If neither side is licked, it falls back to `trial_instruction` to assign a binary left/right label.

ii.
```python
def first_post_go_choice(left_licks: np.ndarray, right_licks: np.ndarray, go_time: float, stop_time: float, instruction: str):
    left_idx = np.searchsorted(left_licks, go_time, side="left")
    right_idx = np.searchsorted(right_licks, go_time, side="left")
    ...
    if np.isfinite(left_time):
        return 0, False
    if np.isfinite(right_time):
        return 1, False
    return (0 if instruction == "left" else 1), True
```

```python
left_licks = np.asarray(f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()], dtype=np.float64)
right_licks = np.asarray(f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()], dtype=np.float64)
...
choice_code, choice_used_fallback = first_post_go_choice(
    left_licks,
    right_licks,
    go_abs,
    float(trial_stop[trial_idx]),
    trial_instruction[trial_idx],
)
```

iii. `CONVERSION_NOTES.md` says the AI wanted `choice` to be determined from actual behavior, i.e. the first post-go lick side, and then use the instructed side only as a fallback for ignore trials because it believed the target format required a binary label. The trajectory explicitly says the AI checked lick-event streams to avoid “silently substituting the instructed side.”

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as binary `0 = left`, `1 = right`, repeats that code across all 80 bins, and does not include a third “no lick” class. If no post-go lick is found before trial stop, it assigns the instructed side as the label.

ii.
```python
if np.isfinite(left_time) and np.isfinite(right_time):
    return (0, False) if left_time <= right_time else (1, False)
...
return (0 if instruction == "left" else 1), True
```

```python
output_trials.append(
    np.vstack(
        [
            np.full(BIN_CENTERS.shape[0], choice_codes[row_idx], dtype=np.int8),
            np.full(BIN_CENTERS.shape[0], outcome_codes[row_idx], dtype=np.int8),
            np.full(BIN_CENTERS.shape[0], early_codes[row_idx], dtype=np.int8),
            tongue_cat,
        ]
    )
)
```

```python
"output_values": [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["lt_40pct", "p40_to_p60", "gt_60pct"],
],
```

iii. `CONVERSION_NOTES.md` explicitly documents this fallback and reports it was used on 13,064 full-dataset trials. The justification given is that the target format was interpreted as requiring a binary left/right choice output.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is read directly from the trial table’s `outcome` field.

ii.
```python
outcome = decode_bytes_array(trials["outcome"][()])[:ephys_trial_count]
...
if outcome[trial_idx] == "ignore":
    outcome_code = 0
elif outcome[trial_idx] == "miss":
    outcome_code = 1
elif outcome[trial_idx] == "hit":
    outcome_code = 2
```

iii. `CONVERSION_NOTES.md` lists the same direct mapping and treats outcome as a required per-trial decoder output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps string outcomes to integer codes `ignore = 0`, `miss = 1`, `hit = 2`, then repeats the per-trial code across all 80 time bins.

ii.
```python
if outcome[trial_idx] == "ignore":
    outcome_code = 0
    stats["ignore_trials_kept"] += 1
elif outcome[trial_idx] == "miss":
    outcome_code = 1
    stats["miss_trials_kept"] += 1
elif outcome[trial_idx] == "hit":
    outcome_code = 2
    stats["hit_trials_kept"] += 1
```

```python
np.full(BIN_CENTERS.shape[0], outcome_codes[row_idx], dtype=np.int8)
```

iii. `CONVERSION_NOTES.md` documents exactly this mapping.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is read directly from the trial table’s `early_lick` field.

ii.
```python
early_lick = decode_bytes_array(trials["early_lick"][()])[:ephys_trial_count]
...
early_code = 1 if early_lick[trial_idx] == "early" else 0
```

iii. `CONVERSION_NOTES.md` says early lick is a per-trial required output and uses the stored trial-table label directly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI converts `early` to `1` and everything else to `0`, then repeats that per-trial value across the 80 time bins.

ii.
```python
early_code = 1 if early_lick[trial_idx] == "early" else 0
early_codes.append(early_code)
```

```python
np.full(BIN_CENTERS.shape[0], early_codes[row_idx], dtype=np.int8)
```

iii. `CONVERSION_NOTES.md` documents the same `no = 0`, `yes = 1` mapping.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue position from `Camera0_side_TongueTracking` timestamps and the second column of its `data` array, which it treats as `tongue_y`. It does not use the likelihood column in downstream processing.

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

iii. `CONVERSION_NOTES.md` explicitly says it used the `tongue_y` coordinate only.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI computes the 40th and 60th percentiles from all raw `tongue_y` values over the full session. For each neural bin within a trial, it takes the last video frame inside the bin; if no frame falls inside the bin, it uses the most recent frame at or before the bin end. It then thresholds those per-bin values into three categories.

ii.
```python
tongue_p40, tongue_p60 = np.percentile(tongue_y, [40.0, 60.0])
```

```python
def trial_tongue_categories(
    video_timestamps: np.ndarray,
    tongue_y: np.ndarray,
    abs_edges: np.ndarray,
    p40: float,
    p60: float,
) -> np.ndarray:
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

iii. `CONVERSION_NOTES.md` gives the same rationale and explicitly says percentiles were computed over the session’s raw `tongue_y` values, with the last frame in each neural bin used as the sampled value.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses session-specific thresholds `p40` and `p60` from the raw session-wide `tongue_y` distribution. Values above `p60` become class `2`, values in `[p40, p60]` become class `1`, and all remaining values stay class `0`. There is no extra class for bins with no visible tongue.

ii.
```python
cats = np.zeros(values.shape[0], dtype=np.int8)
cats[values > p60] = 2
mid = (values >= p40) & (values <= p60)
cats[mid] = 1
return cats
```

```python
"output_values": [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["lt_40pct", "p40_to_p60", "gt_60pct"],
],
```

iii. `CONVERSION_NOTES.md` states the same three-class discretization and says it computed percentiles over the full session’s raw `tongue_y` values.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue output on the same go-cue-centered absolute bin edges used for neural data. For each trial it computes `abs_edges = go + BIN_EDGES` and uses `searchsorted` on camera timestamps to sample one tongue value per neural bin. Trials whose full window falls outside the available camera timestamps are removed.

ii.
```python
abs_edges = go_abs + BIN_EDGES

if abs_edges[0] < video_timestamps[0] or abs_edges[-1] > video_timestamps[-1]:
    stats["n_trials_dropped_window"] += 1
    continue
```

```python
abs_edge_matrix = go_abs[:, None] + BIN_EDGES[None, :]
...
tongue_cat = trial_tongue_categories(
    video_timestamps=video_timestamps,
    tongue_y=tongue_y,
    abs_edges=abs_edge_matrix[row_idx],
    p40=tongue_p40,
    p60=tongue_p60,
)
```

iii. `CONVERSION_NOTES.md` says the tongue output is sampled “for each neural bin,” and the added video-window exclusion reflects the AI’s decision that trials lacking the full aligned camera range should be dropped rather than represented with a missing/hidden category.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases explicitly. Text-like HDF5 fields are decoded with fallback stringification, and optional float fields such as photostimulation metadata map `"N/A"`/`"nan"` to `NaN`. Sessions with no `good` units are dropped. Missing tone times would fall back to `go_time - 1.85`, and missing post-go licks fall back to the instructed side for `choice`. Trials with incomplete side-camera coverage or all-zero neural activity are dropped rather than preserved with missing markers.

ii.
```python
def decode_bytes_array(arr) -> np.ndarray:
    ...
    else:
        out.append(str(value))
```

```python
if value in {"N/A", "nan", "NaN", ""}:
    out[i] = np.nan
```

```python
if good_unit_idx.size == 0:
    return None, stats
```

```python
if hi > lo:
    return float(sample_starts[hi - 1]), False
return float(go_time - 1.85), True
```

```python
if np.isfinite(left_time):
    return 0, False
if np.isfinite(right_time):
    return 1, False
return (0 if instruction == "left" else 1), True
```

iii. The trajectory shows the AI deliberately wanted all fallback decisions to be “auditable in the notes.” `CONVERSION_NOTES.md` documents the tone fallback count, the choice fallback count, and the rationale for excluding trials with incomplete video coverage or all-zero neural activity.

## 10-a. What are the most time-consuming steps of the code?

i. The heaviest work is session-level file I/O plus the per-unit neural binning loop. Each session reads large event/video arrays and ragged spike trains, then performs `searchsorted` and `np.diff` for every retained unit across all trial bin edges. The per-trial tongue sampling loop is another notable cost, but likely secondary to spike processing.

ii.
```python
with h5py.File(path, "r") as f:
    ...
    video_timestamps = np.asarray(...)
    tongue_data = np.asarray(...)
```

```python
for unit_row, unit_idx in enumerate(good_unit_idx):
    spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
    edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
    counts = np.diff(edge_idx, axis=1)
    rates = (counts.astype(np.float32) / BIN_WIDTH).astype(np.float16)
```

```python
for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
    if end_idx > start_idx:
        values[i] = tongue_y[end_idx - 1]
```

iii. The code structure itself shows the dominant cost centers: loading full HDF5 datasets, iterating across all good units, and sampling tongue bins trial by trial. The notes emphasize conversion/runtime summaries rather than micro-optimization, which is consistent with this structure.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain vectorizable: the trial-filter/label-building loop over `keep_idx`, the per-bin tongue frame loop inside `trial_tongue_categories`, the per-trial assembly loop that stacks inputs and outputs, and the inner `for trial_row` loop that copies each unit’s rates into per-trial arrays.

ii.
```python
for trial_idx in keep_idx:
    ...
```

```python
for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
    ...
```

```python
for row_idx, trial_idx in enumerate(valid_trials):
    ...
    input_trials.append(...)
    output_trials.append(...)
```

```python
for unit_row, unit_idx in enumerate(good_unit_idx):
    ...
    for trial_row in range(n_trials):
        neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

iii. The AI did not justify these as intentionally unvectorized in the notes; they are simply how the implementation was written. Relative to the rest of the code, the most obviously avoidable repeated loop is the inner copy from `rates` into `neural_trials`.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly rebuilds trial-level arrays and repeats per-trial scalar values across all 80 bins. It also repeats `searchsorted`-based tongue sampling independently for every trial and copies each unit’s binned rates into trial arrays one trial at a time after already computing `rates` as an `(n_trials, n_bins)` matrix.

ii.
```python
for row_idx, trial_idx in enumerate(valid_trials):
    ...
    input_trials.append(np.vstack([time_from_tone, stim_on]).astype(np.float32, copy=False))
    ...
    output_trials.append(
        np.vstack(
            [
                np.full(BIN_CENTERS.shape[0], choice_codes[row_idx], dtype=np.int8),
                np.full(BIN_CENTERS.shape[0], outcome_codes[row_idx], dtype=np.int8),
                np.full(BIN_CENTERS.shape[0], early_codes[row_idx], dtype=np.int8),
                tongue_cat,
            ]
        )
    )
```

```python
for unit_row, unit_idx in enumerate(good_unit_idx):
    ...
    rates = (counts.astype(np.float32) / BIN_WIDTH).astype(np.float16)
    for trial_row in range(n_trials):
        neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

iii. This repeated work is not called out as a design choice in the notes; it follows from the list-of-trials output format and the AI’s chosen implementation. The main explicit repetition the AI acknowledged elsewhere was that fallback decisions were counted and summarized for auditing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code performs extra bookkeeping that does not affect the final converted dataset used by the decoder. Examples include the unused `BIN_STRIDE` constant, the unused `session_output_counts` accumulator, per-session `trial_indices_source` and `tongue_percentiles` fields that are kept only in the temporary `session` dict, and the full `summary` histogram/statistics pass that is only printed to stdout.

ii.
```python
BIN_STRIDE = 0.05
```

```python
session_output_counts = Counter()
...
session_output_counts[f"outcome_{outcome[trial_idx]}"] += 1
```

```python
session = {
    ...
    "trial_indices_source": valid_trials,
    "tongue_percentiles": (float(tongue_p40), float(tongue_p60)),
}
```

```python
summary = summarize_conversion(data, sessions, session_stats)
return data, summary
```

iii. The AI’s notes emphasize traceability and reporting, so some redundant work appears to have been intentional for documentation and sanity-check purposes rather than for the decoder itself. Still, those pieces are downstream-discarded relative to the final model inputs/outputs.
