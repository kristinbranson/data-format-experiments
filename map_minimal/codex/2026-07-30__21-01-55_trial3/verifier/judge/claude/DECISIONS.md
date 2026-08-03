# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is stored as one NWB file per session under `data/sub-<subject_id>/`. All sessions are found by globbing `sub-*/*.nwb`, sorted alphabetically. Each file is opened with `h5py` (not `pynwb`) and the relevant groups (`units`, `intervals/trials`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`) are read directly from the HDF5 structure.

ii.
```python
for path in sorted(data_dir.glob("sub-*/*.nwb")):
    session, stats = build_session(path, trial_limit=trial_limit)
```

```python
with h5py.File(path, "r") as f:
    units = f["units"]
    trials = f["intervals/trials"]
    go_times = np.asarray(f["acquisition/BehavioralEvents/go_start_times/timestamps"][()], dtype=np.float64)[:ephys_trial_count]
    sample_starts = np.asarray(f["acquisition/BehavioralEvents/sample_start_times/timestamps"][()], dtype=np.float64)
```

iii. From CONVERSION_NOTES.md: "Convert the provided NWB sessions into the decoder-ready pickle format expected by `train_decoder.py`". The AI chose `h5py` over `pynwb` for direct HDF5 access. The glob pattern matches the NWB file layout.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the directory name by stripping the `sub-` prefix from the parent folder of each NWB file (e.g., `sub-440956` becomes `440956`). Unique subjects are sorted and indexed.

ii.
```python
"subject": path.parent.name.replace("sub-", ""),
```

```python
subjects = sorted({s["subject"] for s in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI used the directory naming convention rather than reading `nwb.subject.subject_id` from inside the file. Both yield the same numeric IDs.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session. Each session is identified by the file stem (e.g., `sub-440958_ses-20190216T162508_behavior+ecephys+ogen`). Session order follows the sorted file list.

ii.
```python
for path in sorted(data_dir.glob("sub-*/*.nwb")):
    session, stats = build_session(path, trial_limit=trial_limit)
```

```python
"session_id": path.stem,
```

iii. From CONVERSION_NOTES.md: "Included every NWB session with at least one unit whose `units/classification == 'good'`. This yielded 173 included sessions and 1 skipped session."

## 1-d. How are the data split into trials?

i. Trials come from the `intervals/trials` table in each NWB file. Crucially, the AI truncates all trial-level arrays to match the number of columns in `units/is_good_trials`, reasoning that extra behavioral trials beyond the ephys coverage would produce all-zero neural data.

ii.
```python
ephys_trial_count = int(units["is_good_trials"].shape[1])
trial_start = np.asarray(trials["start_time"][()], dtype=np.float64)[:ephys_trial_count]
trial_stop = np.asarray(trials["stop_time"][()], dtype=np.float64)[:ephys_trial_count]
trial_instruction = decode_bytes_array(trials["trial_instruction"][()])[:ephys_trial_count]
# ... all other trial arrays similarly truncated
go_times = np.asarray(f["acquisition/BehavioralEvents/go_start_times/timestamps"][()], dtype=np.float64)[:ephys_trial_count]
```

iii. From CONVERSION_NOTES.md: "Some NWBs contain more rows in the trial table than are covered by the ephys trial matrix `units/is_good_trials`. To avoid misalignment, all trial-level behavioral/event arrays were truncated to `ephys_trial_count = units['is_good_trials'].shape[1]`."

## 1-e. How are trials filtered based on quality controls?

i. Three filtering steps are applied sequentially:
1. Exclude `auto_water` and `free_water` trials.
2. Drop trials whose aligned `[-2.5, 1.5)` window falls outside the side-camera timestamps.
3. Drop trials that produce all-zero neural activity after binning.

A session is dropped if fewer than 2 trials survive.

ii.
```python
keep_mask = (auto_water == 0) & (free_water == 0)
```

```python
if abs_edges[0] < video_timestamps[0] or abs_edges[-1] > video_timestamps[-1]:
    stats["n_trials_dropped_window"] += 1
    continue
```

```python
nonzero_mask = np.asarray([np.any(trial != 0) for trial in neural_trials], dtype=bool)
if not np.all(nonzero_mask):
    neural_trials = [trial for trial, keep in zip(neural_trials, nonzero_mask) if keep]
```

iii. From CONVERSION_NOTES.md: "The conversion excludes `auto_water` and `free_water` trials... keeps early-lick trials, keeps ignore trials, keeps photostimulation trials." Also: "trials whose aligned `[-2.5, 1.5)` window falls outside side-camera timestamps" and "trials that still produce all-zero neural activity after binning." Total: 88,654 trials kept across 173 sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (ragged spike time arrays per unit) and `units/spike_times_index` (the index into the ragged array). Go cue times from `BehavioralEvents/go_start_times/timestamps` are used to place bin edges.

ii.
```python
spike_times_ds = units["spike_times"]
spike_times_index_ds = units["spike_times_index"]
```

```python
def ragged_row(data_ds, index_ds, row_idx: int) -> np.ndarray:
    stop = int(index_ds[row_idx])
    start = 0 if row_idx == 0 else int(index_ds[row_idx - 1])
    return np.asarray(data_ds[start:stop], dtype=np.float64)
```

iii. Spike times are the only neural representation in the NWB files, so firing rates must be computed from them.

## 2-b. How is the `neural` data processed?

i. For each good unit, spike times are read as a ragged row. Absolute bin edges are computed for all trials at once (go_time + relative_edges). `np.searchsorted` gives running spike counts at each edge; differencing gives per-bin spike counts. Counts are divided by bin width (0.05s) to get firing rates in Hz. The result is stored as `float16`.

ii.
```python
flat_abs_edges = abs_edge_matrix.reshape(-1)
for unit_row, unit_idx in enumerate(good_unit_idx):
    spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
    edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
    counts = np.diff(edge_idx, axis=1)
    rates = (counts.astype(np.float32) / BIN_WIDTH).astype(np.float16)
    for trial_row in range(n_trials):
        neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

iii. From CONVERSION_NOTES.md: "Neural activity is stored as firing rate in spikes/s: `rate = spike_count_in_bin / 0.05`"

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == "good"` are retained. No additional metric thresholds are applied. A session with zero good units is dropped entirely (1 session dropped).

ii.
```python
classification = decode_bytes_array(units["classification"][()])
good_unit_idx = np.flatnonzero(classification == "good")
if good_unit_idx.size == 0:
    return None, stats
```

iii. From CONVERSION_NOTES.md: "Used the NWB-provided quality-control label `classification == 'good'`. This matches the papers' and white paper's description that downstream analyses use units labeled `good` by the QC classifier. No extra firing-rate or waveform filtering was added."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are computed in absolute time by adding the relative bin edges to each trial's go cue time. Spike times and event times share the same session-absolute clock, so no additional alignment step is needed.

ii.
```python
abs_edge_matrix = go_abs[:, None] + BIN_EDGES[None, :]
flat_abs_edges = abs_edge_matrix.reshape(-1)
edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
```

iii. The NWB file stores everything on one global clock, so alignment only requires looking up each trial's go cue time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins spanning [-2.5, 1.5) seconds relative to the go cue, yielding 80 time bins per trial. No rebinning is applied; spikes are binned directly from raw spike times.

ii.
```python
WINDOW_START = -2.5
WINDOW_END = 1.5
BIN_WIDTH = 0.05
BIN_STRIDE = 0.05

BIN_EDGES = fixed_bin_edges(WINDOW_START, WINDOW_END, BIN_WIDTH)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2.0
```

iii. Matches the instructions: "Use 50-ms-width bins for computing firing rates" and "Extract 2.5 s before to 1.5 s after the go cue for each trial."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` timestamps (tone onsets) and go cue times. The tone for each trial is the last `sample_start_times` event between the trial start and the go cue.

ii.
```python
sample_starts = np.asarray(f["acquisition/BehavioralEvents/sample_start_times/timestamps"][()], dtype=np.float64)
```

```python
def last_sample_before_go(sample_starts, trial_start, go_time):
    lo = np.searchsorted(sample_starts, trial_start, side="left")
    hi = np.searchsorted(sample_starts, go_time, side="right")
    if hi > lo:
        return float(sample_starts[hi - 1]), False
    return float(go_time - 1.85), True
```

iii. From CONVERSION_NOTES.md: "The tone onset was taken as the last `sample_start_times` timestamp between trial start and go cue. This follows the task structure in which multiple sample starts can occur due to replay after early licking."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the tone onset is expressed relative to the go cue, then bin centers minus that offset give time from tone onset. A fallback of `go_time - 1.85` is used if no sample_start is found between trial start and go cue (never triggered in practice).

ii.
```python
tone_on_rel = tone_abs[row_idx] - go_abs[row_idx]
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
```

iii. From CONVERSION_NOTES.md: "For each neural bin, the value is: `bin_center_relative_to_go - tone_onset_relative_to_go`. That makes zero correspond to tone onset."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same bin centers used for the neural data are used to compute the time-from-tone values, ensuring perfect alignment.

ii.
```python
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
```

iii. Both neural and input share the same `BIN_CENTERS` array defined relative to the go cue.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset`, `photostim_duration`, and `photostim_power` in the trials table, plus `start_time` and go cue time for coordinate conversion.

ii.
```python
stim_onset = parse_optional_float_array(trials["photostim_onset"][()])[:ephys_trial_count]
stim_duration = parse_optional_float_array(trials["photostim_duration"][()])[:ephys_trial_count]
stim_power = parse_optional_float_array(trials["photostim_power"][()])[:ephys_trial_count]
```

iii. The AI additionally checks that `photostim_power > 0` and all three fields are finite before marking a trial as stimulated.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series: 1 where the bin center falls within [onset, offset) of the stimulation, 0 otherwise. Onset is converted from trial-start-relative to go-cue-relative coordinates. Trials without stimulation (NaN onset/duration or zero power) get all zeros.

ii.
```python
if np.isfinite(stim_power[trial_idx]) and stim_power[trial_idx] > 0 and np.isfinite(stim_onset[trial_idx]) and np.isfinite(stim_duration[trial_idx]):
    go_rel = go_abs - float(trial_start[trial_idx])
    on_rel = float(stim_onset[trial_idx] - go_rel)
    off_rel = on_rel + float(stim_duration[trial_idx])
else:
    on_rel = np.nan
    off_rel = np.nan
```

```python
if np.isfinite(stim_on_rel[row_idx]):
    stim_on = ((BIN_CENTERS >= stim_on_rel[row_idx]) & (BIN_CENTERS < stim_off_rel[row_idx])).astype(np.float32)
else:
    stim_on = np.zeros(BIN_CENTERS.shape[0], dtype=np.float32)
```

iii. From CONVERSION_NOTES.md: "Stimulation times are converted from trial-start reference into go-cue reference... Bins are marked 1 when the bin center lies within the stimulation interval."

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The onset and offset are expressed relative to the go cue, which is the same reference used for the neural bin edges. Bin centers are compared directly against the stimulation interval.

ii.
```python
go_rel = go_abs - float(trial_start[trial_idx])
on_rel = float(stim_onset[trial_idx] - go_rel)
```

iii. Same bin centers as neural data ensure alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from the actual lick timestamps: `left_lick_times` and `right_lick_times` from `BehavioralEvents`. The first post-go lick within the trial window determines the choice. For ignore trials (no lick), the instructed side from `trial_instruction` is used as a fallback.

ii.
```python
left_licks = np.asarray(f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()], dtype=np.float64)
right_licks = np.asarray(f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()], dtype=np.float64)
```

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

iii. From CONVERSION_NOTES.md: "Choice is the first post-go lick side within the trial... Ignore trials have no post-go lick by definition, so they need a fallback to satisfy the required binary label format. For those trials: fallback choice = instructed side from `trial_instruction`."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Binary encoding: left = 0, right = 1. Only two categories (no "no lick" category). For ignore trials, the instructed side is assigned as the choice. The value is expanded across all 80 time bins.

ii.
```python
output_trials.append(
    np.vstack([
        np.full(BIN_CENTERS.shape[0], choice_codes[row_idx], dtype=np.int8),
        ...
    ])
)
```

```python
"output_values": [
    ["left", "right"],
    ...
],
```

iii. From CONVERSION_NOTES.md: "This fallback was used on 13,064 full-dataset trials."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which holds the strings `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcome = decode_bytes_array(trials["outcome"][()])[:ephys_trial_count]
```

iii. The trials table stores the outcome explicitly with the three categories needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: ignore = 0, miss = 1, hit = 2. Expanded across all 80 time bins per trial.

ii.
```python
if outcome[trial_idx] == "ignore":
    outcome_code = 0
elif outcome[trial_idx] == "miss":
    outcome_code = 1
elif outcome[trial_idx] == "hit":
    outcome_code = 2
```

```python
np.full(BIN_CENTERS.shape[0], outcome_codes[row_idx], dtype=np.int8),
```

iii. Matches the instructions: "ignore = 0, miss = 1, hit = 2".

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds strings `'no early'` and `'early'`.

ii.
```python
early_lick = decode_bytes_array(trials["early_lick"][()])[:ephys_trial_count]
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to integers: no = 0, yes = 1. Expanded across all 80 time bins.

ii.
```python
early_code = 1 if early_lick[trial_idx] == "early" else 0
```

```python
np.full(BIN_CENTERS.shape[0], early_codes[row_idx], dtype=np.int8),
```

iii. Matches the instructions: "no = 0, yes = 1".

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which has `data` of shape `(n_frames, 3)` = `(tongue_x, tongue_y, tongue_likelihood)` and matching `timestamps`. Column 1 (`tongue_y`) is the value used.

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

iii. This is the only tongue measurement in the file.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each neural bin, the **last video frame** within that bin is used (not an average). No likelihood filtering is applied -- all raw tongue_y values are used regardless of whether the tongue is visible. Session-wide percentiles (40th, 60th) are computed over ALL raw `tongue_y` values (not just visible frames, not binned means).

ii.
```python
tongue_p40, tongue_p60 = np.percentile(tongue_y, [40.0, 60.0])
```

```python
def trial_tongue_categories(video_timestamps, tongue_y, abs_edges, p40, p60):
    starts = np.searchsorted(video_timestamps, abs_edges[:-1], side="left")
    ends = np.searchsorted(video_timestamps, abs_edges[1:], side="left")
    for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
        if end_idx > start_idx:
            values[i] = tongue_y[end_idx - 1]   # last frame in bin
        else:
            fallback_idx = max(0, min(len(tongue_y) - 1, end_idx - 1))
            values[i] = tongue_y[fallback_idx]   # nearest frame
    cats = np.zeros(values.shape[0], dtype=np.int8)
    cats[values > p60] = 2
    mid = (values >= p40) & (values <= p60)
    cats[mid] = 1
    return cats
```

iii. From CONVERSION_NOTES.md: "For each neural bin, took the last video frame within that bin. If a bin had no new frame, used the most recent available frame at or before the bin end. Discretization is session-specific... Percentiles were computed over the full session's raw `tongue_y` values."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories only:
- 0: < 40th percentile
- 1: 40th to 60th percentile
- 2: > 60th percentile

There is no "not visible" category. Bins with no frames get a fallback to the nearest frame value.

ii.
```python
cats = np.zeros(values.shape[0], dtype=np.int8)
cats[values > p60] = 2
mid = (values >= p40) & (values <= p60)
cats[mid] = 1
```

```python
"output_values": [
    ...
    ["lt_40pct", "p40_to_p60", "gt_60pct"],
],
```

iii. From CONVERSION_NOTES.md: the 40th/60th percentile split follows the instructions.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Video timestamps are searched against the absolute bin edges (go_time + relative edges) using `searchsorted`. The last frame within each bin's time window is selected. If no frame falls in a bin, the nearest preceding frame is used.

ii.
```python
starts = np.searchsorted(video_timestamps, abs_edges[:-1], side="left")
ends = np.searchsorted(video_timestamps, abs_edges[1:], side="left")
```

iii. The camera timestamps share the global clock with the spikes and events, so direct comparison against bin edges provides alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Session with no good units**: Dropped entirely (1 session).
- **Trials beyond ephys coverage**: Truncated via `is_good_trials.shape[1]`.
- **Auto/free water trials**: Excluded.
- **Trials outside video window**: Excluded if the aligned `[-2.5, 1.5)` window extends beyond camera timestamps.
- **All-zero neural trials**: Dropped after binning (127 trials in full dataset).
- **No tone onset found**: Fallback to `go_time - 1.85` (never triggered).
- **Ignore trials with no lick**: Choice falls back to instructed side.
- **Empty tongue bins**: Fallback to nearest preceding frame.

ii.
```python
# All-zero neural trial removal
nonzero_mask = np.asarray([np.any(trial != 0) for trial in neural_trials], dtype=bool)
if not np.all(nonzero_mask):
    neural_trials = [trial for trial, keep in zip(neural_trials, nonzero_mask) if keep]
```

```python
# Tone fallback
return float(go_time - 1.85), True
```

iii. From CONVERSION_NOTES.md: various fallback mechanisms are documented and their usage counts tracked in the summary statistics.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file with `h5py` and extracting the ragged spike time arrays dominate. For each good unit, the full spike time array is read individually via `ragged_row()`. The per-trial loop for tongue categorization and the per-trial loop copying firing rates to output arrays also contribute. The inner loop over trials for copying rates is particularly inefficient:
```python
for trial_row in range(n_trials):
    neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

ii. See code in 2-b.

iii. N/A

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. The per-trial loop copying firing rates to individual trial arrays (lines 301-302).
2. The per-trial input/output construction loop (lines 260-286) that builds each trial's arrays individually.
3. The tongue categorization function `trial_tongue_categories` which loops over bins.
4. The `decode_bytes_array` and `parse_optional_float_array` functions that loop element-by-element.

ii.
```python
for trial_row in range(n_trials):
    neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

```python
for row_idx, trial_idx in enumerate(valid_trials):
    tone_on_rel = tone_abs[row_idx] - go_abs[row_idx]
    time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
    ...
    input_trials.append(np.vstack([time_from_tone, stim_on]).astype(np.float32, copy=False))
```

iii. The reference solution vectorizes input/output construction across all trials at once (e.g., `time_from_tone = CENTERS[None, :] + (go - tone)[:, None]`), avoiding per-trial Python loops.

## 10-c. What processing does the code repeat multiple times?

i. Tongue y percentiles are computed once per session, and bin edges are computed once at module level. However, the per-trial loop recomputes `BIN_CENTERS` operations for each trial independently rather than vectorizing across trials. The `decode_bytes_array` function is called separately for each string column rather than batched.

ii.
```python
for row_idx, trial_idx in enumerate(valid_trials):
    tone_on_rel = tone_abs[row_idx] - go_abs[row_idx]
    time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
```

iii. No truly redundant recomputation, but the per-trial loop structure means similar numpy operations are repeated instead of being vectorized.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads and processes `photostim_power` for determining stimulation status, but the reference solution only checks whether `photostim_onset` is `'N/A'` and does not examine power. The code also reads `auto_water` which the reference does not use. The `trial_indices_source` and `tongue_percentiles` stored per session are metadata not used by the decoder.

ii.
```python
stim_power = parse_optional_float_array(trials["photostim_power"][()])[:ephys_trial_count]
auto_water = np.asarray(trials["auto_water"][()], dtype=np.int8)[:ephys_trial_count]
```

```python
"trial_indices_source": valid_trials,
"tongue_percentiles": (float(tongue_p40), float(tongue_p60)),
```

iii. These extra fields add processing time but do not affect the downstream decoder.
