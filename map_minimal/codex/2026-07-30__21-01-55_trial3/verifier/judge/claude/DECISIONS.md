# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is distributed as one NWB file per session under `data/sub-<subject_id>/`. All sessions are found with a glob pattern (`sub-*/*.nwb`), sorted, and each file is opened with `h5py` (not `pynwb`). Subjects, trials, units, and behavioral events are then read from each HDF5 file's internal structure (`units`, `intervals/trials`, `acquisition/BehavioralEvents`, etc.).

ii.
```python
with h5py.File(path, "r") as f:
    units = f["units"]
    classification = decode_bytes_array(units["classification"][()])
    ...
    trials = f["intervals/trials"]
    trial_start = np.asarray(trials["start_time"][()], dtype=np.float64)[:ephys_trial_count]
    ...
    go_times = np.asarray(f["acquisition/BehavioralEvents/go_start_times/timestamps"][()], dtype=np.float64)[:ephys_trial_count]
```

```python
for path in sorted(data_dir.glob("sub-*/*.nwb")):
    session, stats = build_session(path, trial_limit=trial_limit)
```

iii. The agent initially used `pynwb` for exploration but switched to `h5py` for speed: "The `pynwb` scan is workable for spot checks but too slow for all-session summaries. I'm switching the bulk statistics pass to direct HDF5 reads so I can iterate quickly on session and unit filters without spending minutes on every tweak."

## 1-b. How are the data split into subjects?

i. Subjects are derived from the NWB directory name by stripping the `sub-` prefix (e.g., `sub-440956` -> `'440956'`). The unique sorted set of subject IDs becomes `subjects`, and each session gets an index into that list.

ii.
```python
stats = {
    ...
    "subject": path.parent.name.replace("sub-", ""),
    ...
}
```

```python
subjects = sorted({s["subject"] for s in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The agent derives subjects from the directory name rather than reading `nwb.subject.subject_id`. The numeric IDs are the same either way.

## 1-c. How are the data split into sessions?

i. One NWB file is one session. Each session is identified by the file stem (e.g., `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`). Session order follows the sorted file list.

ii.
```python
for path in sorted(data_dir.glob("sub-*/*.nwb")):
    session, stats = build_session(path, trial_limit=trial_limit)
```

```python
session = {
    "session_id": path.stem,
    ...
}
```

iii. Since the dandiset stores one session per file, the file boundary is the session boundary. Sorting by path gives chronological order within each subject.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`), but are truncated to `ephys_trial_count = units["is_good_trials"].shape[1]`, which represents the number of trials actually covered by ephys recording. The go cue times are similarly truncated.

ii.
```python
ephys_trial_count = int(units["is_good_trials"].shape[1])
trial_start = np.asarray(trials["start_time"][()], dtype=np.float64)[:ephys_trial_count]
...
go_times = np.asarray(f["acquisition/BehavioralEvents/go_start_times/timestamps"][()], dtype=np.float64)[:ephys_trial_count]
```

iii. The agent discovered that some NWB files contain more behavioral trials than ephys-covered trials: "I found the source of the 'all neural data is zero' warnings: some NWB files contain full-session behavior, but the ephys spike trains only cover an initial block of trials." Using `is_good_trials.shape[1]` restricts to the ephys-covered portion.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in three stages: (1) `auto_water == 0` AND `free_water == 0`, (2) video window coverage check (trial window must fall within video timestamps), and (3) post-binning all-zero neural check. A session is dropped if fewer than 2 trials survive. Early lick, ignore, and photostim trials are kept.

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
    ...
    neural_trials = [trial for trial, keep in zip(neural_trials, nonzero_mask) if keep]
```

iii. The agent stated: "The reference texts confirm the key asymmetry I needed to document: the papers exclude early/no-response trials for most analyses, but this decoder task requires those labels, so the conversion keeps them and only removes `auto_water`/`free_water` plus mechanically invalid trials." The all-zero neural check was added because "The cleanest fix here is to enforce a post-binning sanity check: if an entire trial is zero across every good unit and every 50 ms bin, I'll drop it."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (ragged spike time arrays) for units where `units/classification == 'good'`. The go-cue times (`BehavioralEvents/go_start_times`) provide alignment anchors.

ii.
```python
spike_times_ds = units["spike_times"]
spike_times_index_ds = units["spike_times_index"]
...
for unit_row, unit_idx in enumerate(good_unit_idx):
    spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
```

iii. The agent noted: "The NWB files already carry the paper's derived trial table and QC/unit metadata" and used `spike_times` as the raw neural representation.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms bins spanning -2.5s to +1.5s relative to the go cue (80 bins). For each good unit, `np.searchsorted` is used to count spikes per bin across all trials simultaneously, then counts are divided by bin width (0.05s) to get firing rates in Hz. Rates are stored as `float16`.

ii.
```python
flat_abs_edges = abs_edge_matrix.reshape(-1)
...
for unit_row, unit_idx in enumerate(good_unit_idx):
    spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
    edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
    counts = np.diff(edge_idx, axis=1)
    rates = (counts.astype(np.float32) / BIN_WIDTH).astype(np.float16)
```

iii. No smoothing, normalization, or baseline subtraction is applied — raw firing rates from binned spike counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. No additional metric thresholds are applied. A session with zero good units is dropped entirely.

ii.
```python
classification = decode_bytes_array(units["classification"][()])
good_unit_idx = np.flatnonzero(classification == "good")
if good_unit_idx.size == 0:
    return None, stats
```

iii. The agent confirmed: "No extra firing-rate or waveform filtering was added on top of the NWB `good` label."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All bin edges are computed as offsets from the go cue time for each trial. The go cue time is read from `BehavioralEvents/go_start_times/timestamps`. Since all NWB times are on the same absolute clock, no additional alignment step is needed.

ii.
```python
go_times = np.asarray(f["acquisition/BehavioralEvents/go_start_times/timestamps"][()], dtype=np.float64)[:ephys_trial_count]
...
abs_edge_matrix = go_abs[:, None] + BIN_EDGES[None, :]
```

iii. The agent matched the reference on go-cue alignment: "...while still matching the paper/code on go-cue alignment, binning, and unit QC."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms bins, 80 non-overlapping bins spanning [-2.5, 1.5) seconds relative to the go cue. No temporal rebinning or smoothing is applied. The bin grid is defined once and reused for all trials and sessions.

ii.
```python
WINDOW_START = -2.5
WINDOW_END = 1.5
BIN_WIDTH = 0.05
BIN_STRIDE = 0.05

def fixed_bin_edges(start_time, end_time, width):
    n_bins = int(round((end_time - start_time) / width))
    return start_time + np.arange(n_bins + 1, dtype=np.float64) * width

BIN_EDGES = fixed_bin_edges(WINDOW_START, WINDOW_END, BIN_WIDTH)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2.0
```

iii. The agent initially tried an inclusive convention producing 81 bins, then switched: "I'm switching the converter to literal 50 ms bins over [-2.5, 1.5) so the window matches the task specification."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (tone onset timestamps) in `BehavioralEvents`, together with the go cue time. The last `sample_start_times` timestamp between trial start and go cue is taken as the tone onset.

ii.
```python
sample_starts = np.asarray(f["acquisition/BehavioralEvents/sample_start_times/timestamps"][()], dtype=np.float64)
...
tone_time_abs, tone_used_fallback = last_sample_before_go(sample_starts, float(trial_start[trial_idx]), go_abs)
```

```python
def last_sample_before_go(sample_starts, trial_start, go_time):
    lo = np.searchsorted(sample_starts, trial_start, side="left")
    hi = np.searchsorted(sample_starts, go_time, side="right")
    if hi > lo:
        return float(sample_starts[hi - 1]), False
    return float(go_time - 1.85), True
```

iii. The agent noted the task structure allows multiple sample starts due to replay after early licking, so the last one before the go cue is chosen.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the tone onset is expressed relative to the go cue, then the bin centers (which are relative to the go cue) are offset by this difference. There is a fallback: if no sample_start is found between trial_start and go_time, the tone is assumed at `go_time - 1.85` (the expected sample-to-go delay).

ii.
```python
tone_on_rel = tone_abs[row_idx] - go_abs[row_idx]
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
```

iii. The fallback was tracked and reported as occurring in 0 trials across the full dataset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both the neural data and time_from_tone are computed on the same bin grid (80 bins at 50ms intervals relative to the go cue), so they are inherently aligned — bin k of the input corresponds to the same time interval as bin k of the neural data.

ii.
```python
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
```

iii. N/A

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset`, `photostim_duration`, and `photostim_power` in the trials table, with `start_time` and go cue used to place them on the trial's time axis.

ii.
```python
stim_onset = parse_optional_float_array(trials["photostim_onset"][()])[:ephys_trial_count]
stim_duration = parse_optional_float_array(trials["photostim_duration"][()])[:ephys_trial_count]
stim_power = parse_optional_float_array(trials["photostim_power"][()])[:ephys_trial_count]
```

iii. The onsets are stored relative to trial start and need conversion to go-cue-relative coordinates.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time-varying input. Trials are marked as stimulated only if `stim_power > 0` and onset/duration are finite. The onset/offset are converted from trial-start-relative to go-cue-relative coordinates. Bins where the center falls between onset and offset are set to 1, else 0.

ii.
```python
if np.isfinite(stim_power[trial_idx]) and stim_power[trial_idx] > 0 and np.isfinite(stim_onset[trial_idx]) and np.isfinite(stim_duration[trial_idx]):
    go_rel = go_abs - float(trial_start[trial_idx])
    on_rel = float(stim_onset[trial_idx] - go_rel)
    off_rel = on_rel + float(stim_duration[trial_idx])
    ...
else:
    on_rel = np.nan
    off_rel = np.nan
...
if np.isfinite(stim_on_rel[row_idx]):
    stim_on = ((BIN_CENTERS >= stim_on_rel[row_idx]) & (BIN_CENTERS < stim_off_rel[row_idx])).astype(np.float32)
else:
    stim_on = np.zeros(BIN_CENTERS.shape[0], dtype=np.float32)
```

iii. The agent also checks `stim_power > 0` as an additional guard beyond just checking `N/A` onset values.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The onset and offset are expressed relative to the go cue, matching the neural bin grid. Comparison against bin centers ensures bin-level alignment.

ii.
```python
go_rel = go_abs - float(trial_start[trial_idx])
on_rel = float(stim_onset[trial_idx] - go_rel)
```

iii. N/A

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from the actual lick event streams: `left_lick_times` and `right_lick_times` from `BehavioralEvents`. The first lick after the go cue within the trial determines the choice side. For trials with no post-go lick (ignore trials), the instructed side from `trial_instruction` is used as a fallback.

ii.
```python
left_licks = np.asarray(f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()], dtype=np.float64)
right_licks = np.asarray(f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()], dtype=np.float64)
...
choice_code, choice_used_fallback = first_post_go_choice(left_licks, right_licks, go_abs, float(trial_stop[trial_idx]), trial_instruction[trial_idx])
```

```python
def first_post_go_choice(left_licks, right_licks, go_time, stop_time, instruction):
    ...
    if np.isfinite(left_time) and np.isfinite(right_time):
        return (0, False) if left_time <= right_time else (1, False)
    ...
    return (0 if instruction == "left" else 1), True
```

iii. The agent stated: "Choice is the one ambiguous target in the task: the NWB trial table has `trial_instruction` and `outcome`, but not an explicit per-trial reported side. I'm checking the lick-event streams now so I can label choice from behavior itself rather than silently substituting the instructed side."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The choice is coded as 0 (left) or 1 (right) — only 2 categories. There is no separate "no lick" category. For ignore trials, the fallback to the instructed side means they get left=0 or right=1. The value is repeated across all 80 time bins.

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

iii. The agent justified: "Ignore trials have no post-go lick by definition, so they need a fallback to satisfy the required binary label format."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which contains strings `'ignore'`, `'miss'`, and `'hit'`.

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

iii. The trials table stores the outcome explicitly with the three categories requested by the instructions.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to integers: ignore=0, miss=1, hit=2. The value is repeated across all 80 time bins.

ii.
```python
np.full(BIN_CENTERS.shape[0], outcome_codes[row_idx], dtype=np.int8),
```

iii. Straightforward mapping matching the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds `'no early'` and `'early'`.

ii.
```python
early_lick = decode_bytes_array(trials["early_lick"][()])[:ephys_trial_count]
...
early_code = 1 if early_lick[trial_idx] == "early" else 0
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no) or 1 (yes). Repeated across all 80 time bins.

ii.
```python
early_code = 1 if early_lick[trial_idx] == "early" else 0
...
np.full(BIN_CENTERS.shape[0], early_codes[row_idx], dtype=np.int8),
```

iii. Straightforward binary mapping.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` = `tongue_x`, `tongue_y`, `tongue_likelihood`, with matching `timestamps`. Column 1 (`tongue_y`) is the value used.

ii.
```python
video_timestamps = np.asarray(
    f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps"][()], dtype=np.float64)
tongue_data = np.asarray(
    f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"][()], dtype=np.float64)
tongue_y = tongue_data[:, 1]
```

iii. This is the only tongue measurement in the file.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The 40th and 60th percentiles are computed over ALL frames of `tongue_y` for the entire session (no likelihood/visibility filtering). These percentiles serve as class boundaries. Each bin's value is the last video frame within that bin (not a mean). Values below the 40th percentile get class 0, between 40th-60th get class 1, above 60th get class 2. There is no "not visible" class — every bin receives one of the three categories.

ii.
```python
tongue_p40, tongue_p60 = np.percentile(tongue_y, [40.0, 60.0])
```

```python
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
    cats = np.zeros(values.shape[0], dtype=np.int8)
    cats[values > p60] = 2
    mid = (values >= p40) & (values <= p60)
    cats[mid] = 1
    return cats
```

iii. The agent noted: "Percentiles were computed over the full session's raw `tongue_y` values."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories only (no "not visible" class): 0 = below 40th percentile, 1 = 40th to 60th percentile, 2 = above 60th percentile. The thresholds are the 40th and 60th percentiles of all raw tongue_y values (including invisible frames) for the session. For bins with no video frame, a fallback to the nearest available frame is used rather than assigning a "not visible" class.

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

iii. The agent's `output_values` for tongue_y only has 3 entries, not 4.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps share the same absolute clock as spikes and events. For each trial, `searchsorted` is used to find video frames within each neural bin's time window. The last frame within each bin is used as the representative value.

ii.
```python
starts = np.searchsorted(video_timestamps, abs_edges[:-1], side="left")
ends = np.searchsorted(video_timestamps, abs_edges[1:], side="left")
```

iii. The same bin grid (aligned to go cue) is used for both neural and tongue data, ensuring bin-level alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Sessions with no good units**: `classification` decoded via `decode_bytes_array`, sessions with 0 good units are dropped.
- **Ephys-behavior trial mismatch**: Truncated to `is_good_trials.shape[1]`.
- **auto_water and free_water trials**: Excluded.
- **Trials outside video coverage**: Excluded (window check).
- **All-zero neural trials**: Excluded post-binning.
- **Missing tone onset**: Fallback to `go_time - 1.85`.
- **Missing choice (no lick)**: Fallback to instructed side.
- **Missing video frame in bin**: Fallback to nearest available frame.

ii.
```python
if good_unit_idx.size == 0:
    return None, stats
```

```python
ephys_trial_count = int(units["is_good_trials"].shape[1])
```

```python
nonzero_mask = np.asarray([np.any(trial != 0) for trial in neural_trials], dtype=bool)
```

iii. The agent discussed: "I found the source of the 'all neural data is zero' warnings: some NWB files contain full-session behavior, but the ephys spike trains only cover an initial block of trials."

## 10-a. What are the most time-consuming steps of the code?

i. Reading and processing each NWB session is the dominant cost, particularly reading the spike times buffer and computing firing rates via per-unit `searchsorted`. The agent noted: "I'm past the quick-turn phase; this is now dominated by binning tens of thousands of units."

ii. N/A

iii. The agent chose `h5py` over `pynwb` for speed and used `float16` to manage storage.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops: (1) the per-unit loop for spike binning (`for unit_row, unit_idx in enumerate(good_unit_idx)`) which runs one `searchsorted` per unit across all trials, and (2) the per-trial loop for tongue binning in `trial_tongue_categories`. Additionally, a per-trial loop copies rates into individual trial arrays.

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

iii. The per-unit loop is necessary due to ragged spike time storage. The inner per-trial copy loop could be avoided by storing rates as a 3D array and slicing later.

## 10-c. What processing does the code repeat multiple times?

i. The trial-level loop iterates through each trial individually to compute inputs and outputs, which could be vectorized. The `decode_bytes_array` function is called separately for each string column. However, no major computation is duplicated — each quantity is computed once.

ii. N/A

iii. The per-trial loop structure processes one trial at a time for inputs/outputs rather than vectorizing across trials, but this is a style choice not a true repetition.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads `auto_water` and filters on it in addition to `free_water`, but the reference solution only filters `free_water`. The code also computes and tracks extensive statistics (`session_stats`, `summary`) that are printed but not included in the output pickle. The `tongue_data[:, 0]` (tongue_x) and `tongue_data[:, 2]` (tongue_likelihood) columns are read but not used for filtering or output.

ii.
```python
auto_water = np.asarray(trials["auto_water"][()], dtype=np.int8)[:ephys_trial_count]
...
keep_mask = (auto_water == 0) & (free_water == 0)
```

iii. The auto_water filter removes additional trials beyond what the reference solution does.
