# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is distributed as one NWB file per session under `data/sub-<subject_id>/`. All NWB files are found by walking the sorted directory tree. Each file is opened with `h5py` (not `pynwb`) and processed once. Subjects, trials, units, and behavioral events are read from the HDF5 groups directly.

ii. Finding all files:
```python
def load_sorted_nwb_paths(data_root: str) -> list[str]:
    paths = []
    for subject in sorted(os.listdir(data_root)):
        subject_path = os.path.join(data_root, subject)
        if not os.path.isdir(subject_path):
            continue
        for filename in sorted(os.listdir(subject_path)):
            if filename.endswith(".nwb"):
                paths.append(os.path.join(subject_path, filename))
    return paths
```

Loading one session:
```python
with h5py.File(path, "r") as f:
    classifications = decode_array(f["units/classification"][:]).astype(str)
    ...
    trial_group = f["intervals/trials"]
    behavioral_events = f["acquisition/BehavioralEvents"]
    ...
```

iii. The agent explored the dataset structure and confirmed 174 NWB files for 28 subjects. It chose `h5py` for direct HDF5 access rather than `pynwb`, which is a valid alternative approach to reading NWB files.

## 1-b. How are the data split into subjects?

i. Each session's subject is identified by the parent directory name (e.g., `sub-440956`), extracted from the file path. Unique subjects are collected and sorted to form the `subjects` list.

ii.
```python
def get_subject_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(path))
```

```python
subjects = sorted({session["subject"] for session in session_results})
```

iii. The agent derived the subject identity from the directory structure rather than from the NWB file's internal `subject.subject_id` field. This means subject IDs have the form `sub-440956` rather than just `440956`.

## 1-c. How are the data split into sessions?

i. One NWB file is one session, so no grouping or splitting is needed. Each session is identified by the filename (without extension), extracted from the file path.

ii.
```python
def get_session_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
```

iii. The agent confirmed that each NWB file corresponds to one session. Session order follows sorted directory listing.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trial table (`intervals/trials`), but are filtered to only those covered by the electrophysiology recording. The number of ephys-covered trials is determined from `units/is_good_trials.shape[1]`. The actual mapping is done by matching `obs_intervals` of the first good unit back to the behavioral trial table by exact (start, stop) pairs.

ii.
```python
n_ephys_trials = int(f["units/is_good_trials"].shape[1])
...
obs_intervals = np.asarray(f["units/obs_intervals"][:], dtype=np.float64)
obs_index = np.asarray(f["units/obs_intervals_index"][:], dtype=np.int64)
first_good_unit = int(good_unit_indices[0])
obs_start = 0 if first_good_unit == 0 else int(obs_index[first_good_unit - 1])
obs_stop = int(obs_index[first_good_unit])
first_unit_obs = obs_intervals[obs_start:obs_stop]
...
trial_lookup = {
    (round(float(start), 4), round(float(stop), 4)): idx
    for idx, (start, stop) in enumerate(zip(behavior_trial_start, behavior_trial_stop))
}
trial_indices = []
for start, stop in first_unit_obs:
    key = (round(float(start), 4), round(float(stop), 4))
    ...
    trial_indices.append(trial_lookup[key])
```

iii. The agent discovered that some sessions have more behavioral trials than ephys-covered trials, and developed a matching approach using both start and stop times of obs_intervals to identify which behavioral trials have neural data.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered only based on the `obs_intervals` of the first good unit — only trials that have matching electrophysiology recording intervals are kept. **No filtering is applied for `free_water` trials.** Sessions with zero good units are dropped entirely.

ii.
```python
trial_lookup = {
    (round(float(start), 4), round(float(stop), 4)): idx
    for idx, (start, stop) in enumerate(zip(behavior_trial_start, behavior_trial_stop))
}
trial_indices = []
for start, stop in first_unit_obs:
    key = (round(float(start), 4), round(float(stop), 4))
    if key not in trial_lookup:
        raise ValueError(...)
    trial_indices.append(trial_lookup[key])
```

iii. The agent focused on matching ephys-covered trials but did not identify or filter `free_water` trials, which contain no meaningful spike data. The trajectory shows the agent was aware of trial filtering but focused on the obs_intervals mismatch issue.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (the sorted spike times of each unit) and `units/spike_times_index` (the ragged array index). Only units with `classification == 'good'` contribute.

ii.
```python
spike_times_flat=np.asarray(f["units/spike_times"][:], dtype=np.float64),
spike_times_index=np.asarray(f["units/spike_times_index"][:], dtype=np.int64),
```

iii. The agent identified `spike_times` as the neural data source, consistent with the reference approach.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms bins spanning -2.5 s to +1.5 s relative to the go cue. For each good unit, spikes are assigned to trials via `searchsorted` on trial start times, then binned by computing the relative offset from the go cue and flooring to a bin index. Counts are accumulated with `np.add.at`. Counts are stored as `uint8` and then converted to firing rates in Hz by dividing by the bin width, stored as `float16`.

ii.
```python
counts = np.zeros((n_trials, n_bins, n_good_units), dtype=np.uint8)
...
bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
np.add.at(counts[:, :, good_unit_col], (valid_trial_idx, bin_idx), 1)
...
neural_trials.append((spike_counts[trial_idx].T.astype(np.float16) * (1.0 / BIN_SIZE_S)))
```

iii. The agent chose `uint8` for spike counts (max 255 per bin) and `float16` for firing rates to save memory. The `float16` choice risks precision loss for firing rates, and `uint8` could overflow for high-firing neurons in a 50 ms bin (though unlikely in practice).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. Sessions with no good units are dropped entirely.

ii.
```python
classifications = decode_array(f["units/classification"][:]).astype(str)
good_unit_mask = classifications == "good"
n_good_units = int(np.sum(good_unit_mask))
if n_good_units == 0:
    return None
```

iii. The agent confirmed from the QC white paper and data exploration that `classification == 'good'` is the correct unit quality filter. This matches the reference approach.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to the go cue. For each spike, its trial assignment is determined, and then the relative time from the go cue is computed. Spikes falling within the [-2.5, 1.5) window are binned.

ii.
```python
go_times = assert_one_event_per_trial(go_events, trial_start, trial_stop, "go_start_times")
...
rel_spikes = spikes[valid] - go_times[valid_trial_idx]
in_window = (rel_spikes >= WINDOW_START_S) & (rel_spikes < WINDOW_END_S)
```

iii. The agent correctly identified the go cue as the alignment event per the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins spanning -2.5 s to +1.5 s relative to the go cue, giving 80 time bins. No rebinning is applied — spikes are binned directly from their raw timestamps.

ii.
```python
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
BIN_EDGES_S = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_SIZE_S, BIN_SIZE_S, dtype=np.float64)
BIN_CENTERS_S = BIN_EDGES_S[:-1] + (BIN_SIZE_S / 2.0)
```

iii. The parameters match the instructions exactly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` in the behavioral events, together with the go cue time of each trial. The tone taken for a trial is the last one before its go cue.

ii.
```python
sample_events = np.asarray(behavioral_events["sample_start_times"]["timestamps"][:], dtype=np.float64)
...
sample_start_times = last_event_before_per_trial(sample_events, trial_start, go_times, "sample_start_times")
sample_rel = sample_start_times - go_times
```

iii. The agent identified that early licks can replay the sample epoch, so the last sample onset before the go cue is the correct one.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The relative offset of the tone from the go cue is computed, then the time from tone onset for each bin is `BIN_CENTERS - sample_rel` where `sample_rel = sample_start - go_time`.

ii.
```python
sample_rel = sample_start_times - go_times
time_from_tone = (BIN_CENTERS_S[None, :] - sample_rel[:, None]).astype(np.float32)
```

iii. Since `sample_rel` is negative (tone is before go cue), `BIN_CENTERS - sample_rel = BIN_CENTERS + (go - sample)`, which equals time from tone onset. This is mathematically equivalent to the reference approach.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same bin centers used for neural data are used to compute the time from tone onset, so alignment is inherent.

ii.
```python
time_from_tone = (BIN_CENTERS_S[None, :] - sample_rel[:, None]).astype(np.float32)
```

iii. The shared bin grid ensures alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, with `start_time` and go cue time used to place them on the trial's time axis.

ii.
```python
photostim_onset = decode_array(trial_group["photostim_onset"][:])[trial_indices]
photostim_duration = decode_array(trial_group["photostim_duration"][:])[trial_indices]
```

iii. The agent identified the same source variables as the reference.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The photostimulation onset is converted from trial-relative to go-cue-relative time. A bin is marked as 1 if the bin interval overlaps with the stimulation interval (any overlap, not just bin center), and 0 otherwise.

ii.
```python
rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]
rel_off = rel_on + duration
overlap = (bin_left < rel_off) & (bin_right > rel_on)
photostim[trial_idx] = overlap.astype(np.float32)[0]
```

iii. The agent used an overlap-based approach (any part of the bin overlapping with stim period) rather than a center-based approach (bin center falls within stim period). Both are reasonable.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The onset and offset are expressed relative to the go cue, and bin edges are the same grid used for neural data, so alignment is inherent.

ii.
```python
rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]
rel_off = rel_on + duration
```

iii. N/A

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from the actual lick event timestamps: `left_lick_times` and `right_lick_times` from the behavioral events. The first post-go lick determines choice; if none, the first lick anywhere in the trial is used; if still none, the instructed side is used as a fallback.

ii.
```python
left_lick_times = np.asarray(behavioral_events["left_lick_times"]["timestamps"][:], dtype=np.float64)
right_lick_times = np.asarray(behavioral_events["right_lick_times"]["timestamps"][:], dtype=np.float64)
...
choice, choice_sources = compute_choice_labels(
    trial_start=trial_start, trial_stop=trial_stop, go_times=go_times,
    instructions=instructions, left_lick_times=left_lick_times, right_lick_times=right_lick_times,
)
```

iii. The agent explored the data and found that choice is not stored directly. Rather than deriving it from `trial_instruction` x `outcome`, the agent chose to derive it from the raw lick event timestamps.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as `0` (left) or `1` (right) only — there is **no "no lick" category**. For trials with no lick at all (ignore trials), the instructed side is used as a fallback. The output has only 2 values in `output_values[0]`: `['left', 'right']`.

ii.
```python
def compute_choice_labels(...):
    choice = np.zeros(len(trial_start), dtype=np.int8)
    for trial_idx in range(len(trial_start)):
        ...
        if left_post.size or right_post.size:
            choice[trial_idx] = 0 if left_first < right_first else 1
            continue
        if left_all.size or right_all.size:
            choice[trial_idx] = 0 if left_first < right_first else 1
            continue
        choice[trial_idx] = 0 if instructions[trial_idx] == "left" else 1
    return choice, source_counter
```

```python
OUTPUT_VALUES = [
    ["left", "right"],
    ...
]
```

iii. The agent explicitly chose not to include a "no lick" category, instead falling back to the instructed side. The trajectory shows the agent was aware of the ignore case but decided to force a left/right label for all trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `outcome` column of the trials table, which holds the strings `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcomes_raw = decode_array(trial_group["outcome"][:])[trial_indices].astype(str)
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcomes_raw], dtype=np.int8)
```

iii. The trials table stores outcome explicitly with exactly the three categories required.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to `0` (ignore), `1` (miss), `2` (hit) via a dictionary, and repeated across all 80 bins for the per-trial output.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcomes_raw], dtype=np.int8)
...
np.full(len(BIN_CENTERS_S), outcome[trial_idx], dtype=np.int8),
```

iii. Same coding as the reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds `'no early'` and `'early'`.

ii.
```python
early_raw = decode_array(trial_group["early_lick"][:])[trial_indices].astype(str)
early_map = {"no early": 0, "early": 1}
early = np.array([early_map[x] for x in early_raw], dtype=np.int8)
```

iii. Same source as the reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The two strings are mapped to `0` (no) and `1` (yes) via a dictionary, and repeated across all 80 bins.

ii.
```python
early_map = {"no early": 0, "early": 1}
early = np.array([early_map[x] for x in early_raw], dtype=np.int8)
...
np.full(len(BIN_CENTERS_S), early[trial_idx], dtype=np.int8),
```

iii. Same coding as the reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose `data` is `(n_frames, 3)` = `tongue_x`, `tongue_y`, `tongue_likelihood`, with matching `timestamps`. Column 1 (`tongue_y`) is the value used.

ii.
```python
tongue_group = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_data = np.asarray(tongue_group["data"][:], dtype=np.float64)
tongue_timestamps = np.asarray(tongue_group["timestamps"][:], dtype=np.float64)
tongue_y_raw = tongue_data[:, 1]
```

iii. The agent correctly identified the tongue tracking time series as the data source.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The 40th and 60th percentiles are computed over **all raw tongue y values** (no likelihood filtering). For each trial, the tongue y value is sampled at the last frame within each bin (or the nearest frame if no frame falls in the bin). The sampled value is then discretized into 3 categories based on the percentiles.

ii.
```python
tongue_y_raw = tongue_data[:, 1]
q40 = float(np.percentile(tongue_y_raw, 40))
q60 = float(np.percentile(tongue_y_raw, 60))
...
sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
y_binned = tongue_y[sample_idx]
tongue_disc = np.ones_like(y_binned, dtype=np.int8)
tongue_disc[y_binned < q40] = 0
tongue_disc[y_binned > q60] = 2
```

iii. The agent did not filter frames by likelihood before computing percentiles or binning. This means frames where the tongue is not visible (low likelihood) are included in both the percentile calculation and the per-trial discretization, which mixes tracker noise with real tongue positions.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories only: 0 (below 40th pct), 1 (40th to 60th pct), 2 (above 60th pct). There is **no "not visible" category** (class 3). The `output_values` list only has 3 entries: `["lt_40pct", "p40_to_p60", "gt_60pct"]`.

ii.
```python
tongue_disc = np.ones_like(y_binned, dtype=np.int8)
tongue_disc[y_binned < q40] = 0
tongue_disc[y_binned > q60] = 2
```

```python
OUTPUT_VALUES = [
    ...
    ["lt_40pct", "p40_to_p60", "gt_60pct"],
]
```

iii. The instructions specify a 4th category (3: not visible) for bins with no visible tongue. The agent did not implement this, assigning all bins one of the 3 visible categories regardless of tongue visibility.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The tongue timestamps are used with `searchsorted` to find frames falling within each bin edge of the neural time grid. The last frame within each bin is used as the representative value.

ii.
```python
trial_edges = go_times[:, None] + BIN_EDGES_S[None, :]
left_idx = np.searchsorted(timestamps, trial_edges[:, :-1], side="left")
right_idx = np.searchsorted(timestamps, trial_edges[:, 1:], side="left")
sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
y_binned = tongue_y[sample_idx]
```

iii. The agent aligned tongue data to the same go-cue-relative bin grid as the neural data using searchsorted, which is a valid alignment approach.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Two cases are handled:
- **Session with no good units**: `classification` values that are not strings are decoded, and sessions with zero good units are dropped.
- **Missing tongue frames in bins**: When no frame falls within a bin, the last frame before the bin is used as a fallback (via `np.clip(right_idx - 1, 0, ...)`).

`free_water` trials are not filtered. Tongue likelihood is not used to detect missing/unreliable frames.

ii.
```python
def decode_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return value
```

```python
good_unit_mask = classifications == "good"
n_good_units = int(np.sum(good_unit_mask))
if n_good_units == 0:
    return None
```

```python
sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
```

iii. The agent handled byte-string decoding and zero-good-unit sessions but did not handle free_water trials or low-likelihood tongue frames.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file and loading the spike_times buffer dominates. The per-unit spike binning loop with `np.add.at` and trial assignment via `searchsorted` is also expensive. Pickling the output file is the final expensive step.

ii.
```python
spike_times_flat=np.asarray(f["units/spike_times"][:], dtype=np.float64),
...
for unit_idx, unit_end in enumerate(spike_times_index):
    ...
    np.add.at(counts[:, :, good_unit_col], (valid_trial_idx, bin_idx), 1)
```

iii. The agent noted the conversion took substantial time for all 173 sessions.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Three loops could be vectorized:
1. The per-unit spike binning loop iterates over all units, doing trial assignment and binning for each.
2. The per-trial photostim loop iterates over trials to compute bin overlaps.
3. The per-trial choice computation loop iterates over trials to find first lick events.

ii.
```python
for unit_idx, unit_end in enumerate(spike_times_index):
    ...
```

```python
for trial_idx in range(n_trials):
    onset = as_float_or_none(onset_values[trial_idx])
    ...
```

```python
for trial_idx in range(len(trial_start)):
    ...
```

iii. The photostim and choice loops could be vectorized with array operations. The spike binning loop is harder to vectorize due to ragged spike counts per unit.

## 10-c. What processing does the code repeat multiple times?

i. No major repeated processing. Each NWB file is opened once and all quantities are computed in a single pass. The `build_dataset` function is called twice (once for full data, once for sample data), but the sample reuses already-computed session results.

ii.
```python
full_data, full_summary = build_dataset(session_results)
...
sample_results = session_results[:sample_count]
sample_data, sample_summary = build_dataset(sample_results)
```

iii. The double `build_dataset` call is a minor redundancy since the sample is a subset of the full results, but the assembly step is cheap compared to the per-session conversion.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores extensive statistics (`stats` dict per session) including choice source counts, outcome counts, early counts, tongue percentiles, and fallback counts. These are used for metadata/logging but not for the decoder itself. It also generates a separate `sample_data.pkl` file that is not required by the instructions.

ii.
```python
"stats": {
    "n_trials": int(n_trials),
    "n_good_units": int(len(brain_region_labels)),
    "n_behavior_trials": int(n_behavior_trials),
    "n_ephys_trials": int(n_ephys_trials),
    "stim_trial_count": int(stim_trial_count),
    "choice_counts": dict(choice_counter),
    "choice_sources": dict(choice_sources),
    ...
}
```

```python
write_pickle(args.sample_out, sample_data)
```

iii. The statistics are useful for debugging and verification but are not part of the target data format. The sample output file is extra work not required by the instructions.
