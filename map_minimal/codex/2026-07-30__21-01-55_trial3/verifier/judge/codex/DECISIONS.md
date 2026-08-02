# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script iterates over every NWB file matching `sub-*/*.nwb`, opens each with `h5py`, and reads unit metadata, trial tables, behavioral event timestamps, and side-camera tongue tracking. Trial-level arrays are truncated to the number of ephys-covered trials using `units["is_good_trials"].shape[1]`.

ii. ```python
for path in sorted(data_dir.glob("sub-*/*.nwb")):
    session, stats = build_session(path, trial_limit=trial_limit)
```

```python
with h5py.File(path, "r") as f:
    units = f["units"]
    trials = f["intervals/trials"]
    go_times = np.asarray(f["acquisition/BehavioralEvents/go_start_times/timestamps"][()], dtype=np.float64)[:ephys_trial_count]
```

```python
ephys_trial_count = int(units["is_good_trials"].shape[1])
trial_start = np.asarray(trials["start_time"][()], dtype=np.float64)[:ephys_trial_count]
```

iii. In `CONVERSION_NOTES.md`, the agent says it converted the provided NWB sessions directly and truncated behavioral/trial arrays to the ephys-covered trial count to avoid misalignment from extra behavior-only trial rows.

## 1-b. How are the data split into subjects?

i. Subject IDs are taken from the parent directory name of each NWB file by removing the `sub-` prefix. After all sessions are built, unique subjects are sorted into `subjects`, and each session gets a `subject_idx`.

ii. ```python
"subject": path.parent.name.replace("sub-", ""),
```

```python
subjects = sorted({s["subject"] for s in sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
"subject_idx": np.asarray([subject_to_idx[session["subject"]] for session in sessions], dtype=np.int16),
```

iii. The notes explicitly state that `subjects` come from the NWB subject directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. `build_session()` returns one session dictionary per file, and `convert_dataset()` appends it if it is not `None`. Sessions are skipped if they have zero good units or fewer than two retained trials after filtering.

ii. ```python
for path in sorted(data_dir.glob("sub-*/*.nwb")):
    session, stats = build_session(path, trial_limit=trial_limit)
    session_stats.append(stats)
    if session is not None:
        sessions.append(session)
```

```python
if good_unit_idx.size == 0:
    return None, stats
...
if len(valid_trials) < 2:
    return None, stats
...
if n_trials < 2:
    return None, stats
```

iii. The notes say the converter included every NWB session with at least one `classification == "good"` unit, and skipped the single session with zero good units. The two-trial minimum comes from the decoder format requirement.

## 1-d. How are the data split into trials?

i. Trials are indexed from the trial table and behavioral arrays after truncation to `ephys_trial_count`. Candidate trials are selected by `keep_idx`; retained trials are collected in `valid_trials`. Each retained trial gets one go-cue-aligned input matrix, output matrix, and neuron-by-time neural matrix.

ii. ```python
keep_idx = np.flatnonzero(keep_mask)
valid_trials = []
...
for trial_idx in keep_idx:
    go_abs = float(go_times[trial_idx])
    abs_edges = go_abs + BIN_EDGES
    ...
    valid_trials.append(trial_idx)
```

```python
abs_edge_matrix = go_abs[:, None] + BIN_EDGES[None, :]
...
input_trials.append(...)
output_trials.append(...)
neural_trials = [np.empty((good_unit_idx.size, n_bins), dtype=np.float16) for _ in range(n_trials)]
```

iii. The notes justify this as trial-by-trial go-cue alignment with all trial arrays truncated to the subset that truly has ephys coverage.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by excluding `auto_water` and `free_water` trials. The script keeps early-lick, ignore, and photostimulation trials. It also drops trials whose `[-2.5, 1.5)` aligned window falls outside the side-camera timestamps, and later drops retained trials with all-zero neural matrices.

ii. ```python
keep_mask = (auto_water == 0) & (free_water == 0)
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
```

iii. The notes explain that the reference code’s “regular trial” mask excludes early lick, no-response, and stimulation trials, but the agent kept them because `early_lick`, `outcome`, and `photostim_on` are required decoder targets/inputs. The extra window/all-zero exclusions are described as mechanical sanity filters.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from `units["spike_times"]` and `units["spike_times_index"]` for units with `classification == "good"`. Go-cue timestamps define the per-trial aligned bin edges.

ii. ```python
classification = decode_bytes_array(units["classification"][()])
good_unit_idx = np.flatnonzero(classification == "good")
```

```python
spike_times_ds = units["spike_times"]
spike_times_index_ds = units["spike_times_index"]
go_times = np.asarray(f["acquisition/BehavioralEvents/go_start_times/timestamps"][()], dtype=np.float64)[:ephys_trial_count]
```

iii. The notes say the agent used the NWB “good” label directly because that matches the paper’s and white paper’s QC description.

## 2-b. How is the `neural` data processed?

i. For each good unit, the code reconstructs its ragged spike-time vector, counts spikes in each absolute go-cue-aligned 50 ms bin with `np.searchsorted` and `np.diff`, converts counts to firing rates in spikes/s, and stores the result as `float16`.

ii. ```python
spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
counts = np.diff(edge_idx, axis=1)
rates = (counts.astype(np.float32) / BIN_WIDTH).astype(np.float16)
```

iii. The notes say the converter uses exact-width bins and stores neural activity as firing rate, `spike_count_in_bin / 0.05`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The main QC filter is `classification == "good"` at the unit level. Sessions with zero good units are skipped. After binning, the script also drops trials whose neural matrices are entirely zero. No additional firing-rate or waveform-based unit filtering is applied.

ii. ```python
good_unit_idx = np.flatnonzero(classification == "good")
if good_unit_idx.size == 0:
    return None, stats
```

```python
nonzero_mask = np.asarray([np.any(trial != 0) for trial in neural_trials], dtype=bool)
if not np.all(nonzero_mask):
    ...
```

iii. The notes explicitly say the agent used the NWB-provided `good` label only, because the papers describe downstream analyses as using classifier-labeled good units, and that it did not add extra unit QC beyond that.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to go cue onset. The code defines fixed relative bin edges from `-2.5` to `1.5` s, then shifts those edges by the trial’s absolute go time.

ii. ```python
WINDOW_START = -2.5
WINDOW_END = 1.5
BIN_EDGES = fixed_bin_edges(WINDOW_START, WINDOW_END, BIN_WIDTH)
```

```python
go_abs = float(go_times[trial_idx])
abs_edges = go_abs + BIN_EDGES
```

iii. The notes list go cue onset as the alignment event and `[-2.5, 1.5)` s as the aligned window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50 ms. The code creates exact non-overlapping 50 ms bins over the 4 s window, yielding 80 bins per trial. No further temporal rebinning or smoothing is applied.

ii. ```python
BIN_WIDTH = 0.05
BIN_STRIDE = 0.05
BIN_EDGES = fixed_bin_edges(WINDOW_START, WINDOW_END, BIN_WIDTH)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2.0
```

iii. The notes say the converter uses exact-width bins over `[start, end)` with 80 bins total and no overlapping windows.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times/timestamps`, together with each trial’s `start_time` and go-cue time. The chosen tone time is the last sample-start event between trial start and go cue.

ii. ```python
sample_starts = np.asarray(f["acquisition/BehavioralEvents/sample_start_times/timestamps"][()], dtype=np.float64)
...
tone_time_abs, tone_used_fallback = last_sample_before_go(sample_starts, float(trial_start[trial_idx]), go_abs)
```

iii. The notes say the agent used the last `sample_start_times` timestamp before go cue because sample/delay epochs can replay after early licking.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The script finds the last sample start in the trial before go cue. If none exists, it falls back to `go_time - 1.85`. It then computes a time-varying ramp at the neural bin centers by subtracting the tone-onset-relative-to-go from each bin center.

ii. ```python
def last_sample_before_go(sample_starts: np.ndarray, trial_start: float, go_time: float):
    lo = np.searchsorted(sample_starts, trial_start, side="left")
    hi = np.searchsorted(sample_starts, go_time, side="right")
    if hi > lo:
        return float(sample_starts[hi - 1]), False
    return float(go_time - 1.85), True
```

```python
tone_on_rel = tone_abs[row_idx] - go_abs[row_idx]
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
```

iii. The notes justify the “last sample start before go” rule using replayed sample epochs; they also note that the hard-coded fallback was never actually used (`tone_fallback_trials: 0`).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is sampled at the same 80 go-cue-aligned bin centers as the neural data and stacked into the per-trial input matrix.

ii. ```python
time_from_tone = (BIN_CENTERS - tone_on_rel).astype(np.float32)
...
input_trials.append(np.vstack([time_from_tone, stim_on]).astype(np.float32, copy=False))
```

iii. The notes say this makes zero correspond to tone onset while keeping the same time base as the go-cue-aligned neural bins.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from the trial-table columns `photostim_onset`, `photostim_duration`, and `photostim_power`, together with trial start and go-cue time to convert the timing reference frame.

ii. ```python
stim_onset = parse_optional_float_array(trials["photostim_onset"][()])[:ephys_trial_count]
stim_duration = parse_optional_float_array(trials["photostim_duration"][()])[:ephys_trial_count]
stim_power = parse_optional_float_array(trials["photostim_power"][()])[:ephys_trial_count]
```

iii. The notes say the binary `photostim_on` input comes from onset, duration, and power, with times shifted from trial-start reference into go-cue reference.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The script marks a trial as stimulated only if power is finite and positive and onset/duration are finite. It converts onset/offset into go-cue-relative time and stores `NaN` for non-stimulated trials.

ii. ```python
if np.isfinite(stim_power[trial_idx]) and stim_power[trial_idx] > 0 and np.isfinite(stim_onset[trial_idx]) and np.isfinite(stim_duration[trial_idx]):
    go_rel = go_abs - float(trial_start[trial_idx])
    on_rel = float(stim_onset[trial_idx] - go_rel)
    off_rel = on_rel + float(stim_duration[trial_idx])
else:
    on_rel = np.nan
    off_rel = np.nan
```

iii. The notes say this mirrors the reference code pattern where stimulation times are shifted from trial-start coordinates to go-cue coordinates.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is converted into a binary vector on the same neural bin centers: a bin is `1` if its center falls within `[stim_on_rel, stim_off_rel)`, otherwise `0`.

ii. ```python
if np.isfinite(stim_on_rel[row_idx]):
    stim_on = ((BIN_CENTERS >= stim_on_rel[row_idx]) & (BIN_CENTERS < stim_off_rel[row_idx])).astype(np.float32)
else:
    stim_on = np.zeros(BIN_CENTERS.shape[0], dtype=np.float32)
```

iii. The notes explicitly say bins are marked `1` when the neural bin center lies within the stimulation interval.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from absolute left-lick and right-lick timestamp streams, plus each trial’s stop time. If no post-go lick exists, the code falls back to `trial_instruction`.

ii. ```python
left_licks = np.asarray(f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()], dtype=np.float64)
right_licks = np.asarray(f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()], dtype=np.float64)
trial_instruction = decode_bytes_array(trials["trial_instruction"][()])[:ephys_trial_count]
```

```python
choice_code, choice_used_fallback = first_post_go_choice(
    left_licks,
    right_licks,
    go_abs,
    float(trial_stop[trial_idx]),
    trial_instruction[trial_idx],
)
```

iii. The notes say the agent wanted a true choice label from the first post-go lick, but used instructed side as a fallback on ignore trials because the target format required a binary choice label for every retained trial.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The function `first_post_go_choice()` searches for the earliest left and right licks at or after go cue and before trial stop. If both exist, it takes the earlier one; if only one exists, it uses that side. If neither exists, it returns the instructed side. The final code maps left to `0` and right to `1`, then repeats that per-trial label across all bins.

ii. ```python
def first_post_go_choice(left_licks, right_licks, go_time, stop_time, instruction):
    ...
    if np.isfinite(left_time) and np.isfinite(right_time):
        return (0, False) if left_time <= right_time else (1, False)
    if np.isfinite(left_time):
        return 0, False
    if np.isfinite(right_time):
        return 1, False
    return (0 if instruction == "left" else 1), True
```

```python
np.full(BIN_CENTERS.shape[0], choice_codes[row_idx], dtype=np.int8)
```

iii. The notes justify the fallback as a decoder-format workaround for ignore trials with no post-go lick.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived directly from the trial-table `outcome` field.

ii. ```python
outcome = decode_bytes_array(trials["outcome"][()])[:ephys_trial_count]
```

iii. The notes describe `outcome` as a direct per-trial categorical output from the trial table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The script maps outcome strings to integer class IDs: `ignore -> 0`, `miss -> 1`, `hit -> 2`. It raises an error on any unexpected value.

ii. ```python
if outcome[trial_idx] == "ignore":
    outcome_code = 0
elif outcome[trial_idx] == "miss":
    outcome_code = 1
elif outcome[trial_idx] == "hit":
    outcome_code = 2
else:
    raise ValueError(...)
```

iii. The notes list the same mapping and present `outcome` as a required retained decoder target, including ignore trials.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. There is no `distance to reward zone` variable anywhere in the task or in `convert_data.py`. For the actual `outcome` output, the script broadcasts the per-trial category across all 80 neural time bins.

ii. ```python
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

iii. The notes describe `outcome` as “per-trial, expanded across time bins.” The “distance to reward zone” wording appears to be a leftover from another template rather than an implemented variable.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It is derived directly from the trial-table `early_lick` field.

ii. ```python
early_lick = decode_bytes_array(trials["early_lick"][()])[:ephys_trial_count]
```

iii. The notes say early-lick trials were kept because `early_lick` is itself a required decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The script maps the string `"early"` to `1` and everything else to `0`, then repeats that per-trial label across all bins.

ii. ```python
early_code = 1 if early_lick[trial_idx] == "early" else 0
...
np.full(BIN_CENTERS.shape[0], early_codes[row_idx], dtype=np.int8)
```

iii. The notes describe `early_lick` as a per-trial categorical output with mapping `no = 0`, `yes = 1`.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from the side-camera tongue-tracking time series: `Camera0_side_TongueTracking/data` and its timestamps. The code takes the second column of the data matrix as `tongue_y`.

ii. ```python
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

iii. The notes say the source is side-camera tongue tracking and that only the `tongue_y` coordinate is used.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each neural bin in each trial, the script finds the last video frame that falls within the bin. If a bin has no new frame, it falls back to the most recent available frame at or before the bin end.

ii. ```python
starts = np.searchsorted(video_timestamps, abs_edges[:-1], side="left")
ends = np.searchsorted(video_timestamps, abs_edges[1:], side="left")
...
if end_idx > start_idx:
    values[i] = tongue_y[end_idx - 1]
else:
    fallback_idx = max(0, min(len(tongue_y) - 1, end_idx - 1))
    values[i] = tongue_y[fallback_idx]
```

iii. The notes give the same description: use the last frame within each neural bin, otherwise reuse the most recent frame up to the bin end.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The code computes the 40th and 60th percentiles of raw `tongue_y` over the full session, then assigns categories `0` for values below `p40`, `1` for values between `p40` and `p60` inclusive, and `2` for values above `p60`.

ii. ```python
tongue_p40, tongue_p60 = np.percentile(tongue_y, [40.0, 60.0])
```

```python
cats = np.zeros(values.shape[0], dtype=np.int8)
cats[values > p60] = 2
mid = (values >= p40) & (values <= p60)
cats[mid] = 1
```

iii. The notes say the discretization is session-specific, with percentiles computed over the full session’s raw `tongue_y` values.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The script aligns tongue values to the same per-trial absolute bin edges used for neural spike counts (`go_abs + BIN_EDGES`) and outputs one tongue category per neural bin.

ii. ```python
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

iii. The notes say tongue y-position is sampled per neural bin, so it shares the same 80-bin go-cue-aligned time base as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles missing or inconsistent data with several heuristics: string-valued missing photostim fields are converted to `NaN`; trial-level arrays are truncated to `ephys_trial_count`; if no sample start is found it falls back to `go_time - 1.85`; if no post-go lick is found it falls back to instructed side; if a tongue bin has no new frame it reuses the latest earlier frame; and trials are dropped if the full aligned window is outside the video range or if the neural matrix is all zero.

ii. ```python
if value in {"N/A", "nan", "NaN", ""}:
    out[i] = np.nan
```

```python
ephys_trial_count = int(units["is_good_trials"].shape[1])
```

```python
return float(go_time - 1.85), True
...
return (0 if instruction == "left" else 1), True
```

```python
fallback_idx = max(0, min(len(tongue_y) - 1, end_idx - 1))
...
if abs_edges[0] < video_timestamps[0] or abs_edges[-1] > video_timestamps[-1]:
    ...
```

iii. The notes call the trial truncation a critical correction for misalignment, note that tone fallback was never needed, and justify the choice fallback as necessary to preserve binary choice labels on retained ignore trials.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is constructing neural firing-rate matrices: iterating over every good unit, calling `searchsorted` on all flattened trial edges, and then copying rates back trial by trial. A smaller but still repeated cost is looping through every bin in `trial_tongue_categories()` for every trial.

ii. ```python
for unit_row, unit_idx in enumerate(good_unit_idx):
    spike_times = ragged_row(spike_times_ds, spike_times_index_ds, int(unit_idx))
    edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
    counts = np.diff(edge_idx, axis=1)
    rates = (counts.astype(np.float32) / BIN_WIDTH).astype(np.float16)
    for trial_row in range(n_trials):
        neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

```python
for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
    ...
```

iii. The agent did not explicitly discuss runtime in the notes, but these are the obvious hot loops in `convert_data.py`.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner `for trial_row` write-back loop after `rates` is computed could be removed by storing neural data in a trial-major array and transposing once. The per-bin loop in `trial_tongue_categories()` could also be vectorized with indexed gathers/fills. The byte-decoding and optional-float parsing helper loops are additionally scalarized.

ii. ```python
for trial_row in range(n_trials):
    neural_trials[trial_row][unit_row, :] = rates[trial_row]
```

```python
for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
    ...
```

iii. These opportunities follow directly from the implementation: both loops operate over arrays whose index arithmetic is already prepared in vector form.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly performs per-trial `searchsorted`-style alignment work: once for tone timing, once for tongue frame lookup, and once per unit for spike binning. It also repeatedly expands per-trial categorical outputs into full-length 80-bin vectors.

ii. ```python
tone_time_abs, tone_used_fallback = last_sample_before_go(...)
...
tongue_cat = trial_tongue_categories(...)
...
edge_idx = np.searchsorted(spike_times, flat_abs_edges, side="left").reshape(n_trials, -1)
```

```python
np.full(BIN_CENTERS.shape[0], choice_codes[row_idx], dtype=np.int8),
np.full(BIN_CENTERS.shape[0], outcome_codes[row_idx], dtype=np.int8),
np.full(BIN_CENTERS.shape[0], early_codes[row_idx], dtype=np.int8),
```

iii. The notes do not discuss redundancy, but the code clearly redoes the same fixed-length label expansion for every retained trial and repeats several alignment passes over the same trial structure.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is `session_output_counts`, which is updated on every retained trial and then never used. The code also tracks `n_trials_dropped_tone` in `stats` but never increments it, so that field contributes nothing.

ii. ```python
session_output_counts = Counter()
...
session_output_counts[f"outcome_{outcome[trial_idx]}"] += 1
```

```python
"n_trials_dropped_tone": 0,
```

iii. The agent’s notes do not mention these as intentional outputs, and no later function consumes `session_output_counts` or a tone-drop count.
