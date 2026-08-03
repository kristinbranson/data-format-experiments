# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from `data/sub-*/` directories using `h5py.File()` directly (not `pynwb`). Files are discovered by listing subdirectories sorted alphabetically. Each file is opened and processed independently via `convert_session()`.

ii.
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

```python
with h5py.File(path, "r") as f:
    classifications = decode_array(f["units/classification"][:]).astype(str)
```

iii. The AI chose `h5py` instead of `pynwb` for direct HDF5 access. The CONVERSION_NOTES.md states the NWB files were verified to expose the same variables used by the reference preprocessing code.

## 1-b. How are the data split into subjects?

i. The AI derives the subject identifier from the directory name containing the NWB file (e.g. `sub-440956`), rather than reading `nwb.subject.subject_id` from inside the file.

ii.
```python
def get_subject_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(path))
```

iii. The CONVERSION_NOTES.md does not specifically justify this choice. The AI relies on the DANDI dataset directory structure convention where each subject has its own `sub-<id>` folder.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session ID is derived from the filename (without extension) rather than from `nwb.identifier`.

ii.
```python
def get_session_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
```

iii. Since each NWB file corresponds to one session, the file boundary is the session boundary, consistent with the dataset structure. This matches the reference approach structurally.

## 1-d. How are the data split into trials?

i. Trials come from the NWB intervals/trials table. The AI matches ephys-covered trials to behavioral trials using `obs_intervals` from the first good unit, looking up matching `(start_time, stop_time)` pairs.

ii.
```python
n_ephys_trials = int(f["units/is_good_trials"].shape[1])
trial_group = f["intervals/trials"]
n_behavior_trials = int(len(trial_group["start_time"]))
...
first_unit_obs = obs_intervals[obs_start:obs_stop]
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

iii. The AI notes that for sessions where the behavior table contains more trials than the ephys recording, trials are matched from the first good unit's `obs_intervals` back to the behavioral trial table to avoid including behavior-only trials with no neural recording.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials using `obs_intervals` to keep only ephys-covered trials. However, the AI does **not** filter out `free_water` trials. All ephys-covered trials are retained, including early-lick and ignore trials.

ii.
```python
# obs_intervals matching (see 1-d above)
trial_indices = np.asarray(trial_indices, dtype=np.int64)
trial_start = behavior_trial_start[trial_indices]
trial_stop = behavior_trial_stop[trial_indices]
```

No `free_water` filtering is present in the code.

iii. The CONVERSION_NOTES.md states: "All ephys-covered trials are retained, including photostim, early-lick, and ignore trials. This differs from some reference analysis masks because the requested decoder outputs explicitly require those conditions." However, `free_water` trials are a separate category that the reference excludes because they contain no spikes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (sorted spike times) and `units/spike_times_index` (ragged array index). Only units with `classification == 'good'` are used.

ii.
```python
spike_counts = bin_spike_counts_for_good_units(
    spike_times_flat=np.asarray(f["units/spike_times"][:], dtype=np.float64),
    spike_times_index=np.asarray(f["units/spike_times_index"][:], dtype=np.int64),
    good_unit_mask=good_unit_mask,
    ...
)
```

iii. The CONVERSION_NOTES.md states: "Neural data: spike counts per 50 ms bin, converted to firing rates by dividing by 0.05 s."

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms windows relative to go cue. Spike counts are computed per bin using `np.add.at`, then converted to firing rates (Hz) by dividing by 0.05. The firing rates are stored as `float16`.

ii.
```python
bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
np.add.at(counts[:, :, good_unit_col], (valid_trial_idx, bin_idx), 1)
```

```python
neural_trials.append((spike_counts[trial_idx].T.astype(np.float16) * (1.0 / BIN_SIZE_S)))
```

iii. The AI stores counts in `uint8` (max 255 spikes per bin) and converts to `float16` firing rates. The reference uses `float32`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. Sessions with zero good units are dropped. No individual quality metric thresholds are applied.

ii.
```python
classifications = decode_array(f["units/classification"][:]).astype(str)
good_unit_mask = classifications == "good"
n_good_units = int(np.sum(good_unit_mask))
if n_good_units == 0:
    return None
```

iii. The CONVERSION_NOTES.md states: "Units are filtered with `classification == 'good'`, matching the published classifier-based QC workflow."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to the go cue onset. For each unit, spikes are assigned to trials by `searchsorted` on trial start times, then binned relative to the go cue time into the [-2.5, 1.5] window.

ii.
```python
rel_spikes = spikes[valid] - go_times[valid_trial_idx]
in_window = (rel_spikes >= WINDOW_START_S) & (rel_spikes < WINDOW_END_S)
...
bin_idx = np.floor((rel_spikes - WINDOW_START_S) / BIN_SIZE_S).astype(np.int64)
```

iii. The CONVERSION_NOTES.md states: "Trials are aligned to go cue onset and restricted to the interval [-2.5 s, +1.5 s) using 50 ms bins."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 50 ms, with 80 bins spanning [-2.5 s, +1.5 s) relative to go cue. No rebinning is applied; spikes are directly binned from spike times.

ii.
```python
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
BIN_EDGES_S = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_SIZE_S, BIN_SIZE_S, dtype=np.float64)
```

iii. The 50 ms bin size follows the instructions directly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times` (tone onset events) and go cue times. The last sample event before the go cue is selected as the tone onset for each trial.

ii.
```python
sample_events = np.asarray(behavioral_events["sample_start_times"]["timestamps"][:], dtype=np.float64)
...
sample_start_times = last_event_before_per_trial(sample_events, trial_start, go_times, "sample_start_times")
```

iii. The AI handles the case where early licks can cause replays of the sample epoch by taking the last sample event before the go cue, consistent with the reference.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset is computed as bin centers (relative to go cue) minus the sample-to-go offset.

ii.
```python
sample_rel = sample_start_times - go_times
time_from_tone = (BIN_CENTERS_S[None, :] - sample_rel[:, None]).astype(np.float32)
```

iii. No additional justification beyond the standard computation.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both neural data and time-from-tone use the same bin centers defined relative to the go cue, ensuring alignment.

ii.
```python
BIN_CENTERS_S = BIN_EDGES_S[:-1] + (BIN_SIZE_S / 2.0)
```

iii. N/A

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `photostim_onset` and `photostim_duration` in the trials table, combined with `start_time` and go cue times.

ii.
```python
photostim_onset = decode_array(trial_group["photostim_onset"][:])[trial_indices]
photostim_duration = decode_array(trial_group["photostim_duration"][:])[trial_indices]
```

iii. The CONVERSION_NOTES.md states: "Input photostim_on: binary per bin, 1 if the photostim interval overlaps the bin."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI computes photostim onset and offset relative to go cue, then marks a bin as 1 if any part of the photostim interval overlaps with the bin edges (edge overlap method), rather than checking if the bin center falls within the interval.

ii.
```python
rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]
rel_off = rel_on + duration
overlap = (bin_left < rel_off) & (bin_right > rel_on)
photostim[trial_idx] = overlap.astype(np.float32)[0]
```

iii. The AI uses a bin-edge overlap approach rather than a bin-center comparison. This means a bin is marked as 1 if any portion of the photostim interval overlaps the bin window, which may produce slightly different results at the boundaries.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Photostim onset/offset are expressed relative to the go cue (same reference as neural bins), ensuring alignment.

ii.
```python
rel_on = (trial_start[trial_idx] + onset) - go_times[trial_idx]
```

iii. N/A

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from actual lick event times (`left_lick_times` and `right_lick_times`), with a fallback to `trial_instruction` for trials with no licks. This differs from the reference which derives choice from `trial_instruction` x `outcome`.

ii.
```python
left_lick_times = np.asarray(behavioral_events["left_lick_times"]["timestamps"][:], dtype=np.float64)
right_lick_times = np.asarray(behavioral_events["right_lick_times"]["timestamps"][:], dtype=np.float64)
```

```python
def compute_choice_labels(trial_start, trial_stop, go_times, instructions, left_lick_times, right_lick_times):
    ...
    left_post = left_all[left_all >= go_time]
    right_post = right_all[right_all >= go_time]
    if left_post.size or right_post.size:
        choice[trial_idx] = 0 if left_first < right_first else 1
        ...
    ...
    choice[trial_idx] = 0 if instructions[trial_idx] == "left" else 1
    source_counter["instruction_fallback"] += 1
```

iii. The CONVERSION_NOTES.md states: "Output choice: first post-go lick side when available; otherwise first lick anywhere in the trial; otherwise instructed side if the trial has no licks at all." The AI's `output_values` for choice is `['left', 'right']` (2 values), with no separate "no lick" category.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI uses a hierarchical approach: (1) first post-go lick, (2) first lick anywhere in the trial, (3) instructed side fallback. This always assigns left (0) or right (1), with no "no lick" / "ignore" category. The reference uses instruction x outcome to derive choice, and adds a third value (2) for "no lick" on ignore trials.

ii.
```python
OUTPUT_VALUES = [
    ["left", "right"],  # only 2 values, no "no lick"
    ...
]
```

```python
choice[trial_idx] = 0 if instructions[trial_idx] == "left" else 1
source_counter["instruction_fallback"] += 1
```

iii. The CONVERSION_NOTES.md reports 12,662 no-lick trials requiring instructed-side fallback. This means those trials are assigned a fabricated choice direction rather than being labeled as "no lick."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived directly from the `outcome` column of the trials table, which contains `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcomes_raw = decode_array(trial_group["outcome"][:])[trial_indices].astype(str)
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcomes_raw], dtype=np.int8)
```

iii. Straightforward mapping from the trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three outcome strings are mapped to integers: ignore=0, miss=1, hit=2, and repeated across all 80 bins.

ii.
```python
np.full(len(BIN_CENTERS_S), outcome[trial_idx], dtype=np.int8),
```

iii. Matches the instructions exactly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from the `early_lick` column of the trials table.

ii.
```python
early_raw = decode_array(trial_group["early_lick"][:])[trial_indices].astype(str)
early_map = {"no early": 0, "early": 1}
early = np.array([early_map[x] for x in early_raw], dtype=np.int8)
```

iii. Direct mapping from trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The two strings are mapped to integers: `no early`=0, `early`=1, repeated across all 80 bins.

ii.
```python
np.full(len(BIN_CENTERS_S), early[trial_idx], dtype=np.int8),
```

iii. Matches the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `Camera0_side_TongueTracking` data, specifically column 1 (tongue_y) from the `(n_frames, 3)` array.

ii.
```python
tongue_group = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_data = np.asarray(tongue_group["data"][:], dtype=np.float64)
tongue_timestamps = np.asarray(tongue_group["timestamps"][:], dtype=np.float64)
tongue_y_raw = tongue_data[:, 1]
```

iii. Same source as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI computes percentiles (40th, 60th) on the **raw, unfiltered** tongue y values over the entire session (no likelihood thresholding). For binning, it uses the **last frame** in each 50 ms bin rather than the mean of frames in the bin.

ii.
```python
tongue_y_raw = tongue_data[:, 1]
q40 = float(np.percentile(tongue_y_raw, 40))
q60 = float(np.percentile(tongue_y_raw, 60))
```

```python
sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
y_binned = tongue_y[sample_idx]
```

iii. The CONVERSION_NOTES.md states: "Binned values use the last frame inside each 50 ms bin, matching the reference marker-alignment style." However, the reference solution filters frames by likelihood threshold (< 0.5 set to NaN) and uses mean of frames per bin. The AI does not apply any likelihood filtering.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses 3 categories: below 40th percentile (0), 40th to 60th (1), above 60th (2). There is no "not visible" category. The reference uses 4 categories including "not visible" (3) for bins with no high-likelihood frames.

ii.
```python
OUTPUT_VALUES = [
    ...
    ["lt_40pct", "p40_to_p60", "gt_60pct"],  # only 3 values
]
```

```python
tongue_disc = np.ones_like(y_binned, dtype=np.int8)
tongue_disc[y_binned < q40] = 0
tongue_disc[y_binned > q60] = 2
```

iii. The AI does not account for frames where the tongue is not visible (retracted), always assigning one of the three position categories.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI uses `searchsorted` on camera timestamps with the go-cue-relative bin edges, then takes the last frame in each bin.

ii.
```python
trial_edges = go_times[:, None] + BIN_EDGES_S[None, :]
left_idx = np.searchsorted(timestamps, trial_edges[:, :-1], side="left")
right_idx = np.searchsorted(timestamps, trial_edges[:, 1:], side="left")
sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
```

iii. The AI aligns using the same go-cue-relative bin edges as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases:
- **Sessions with no good units**: dropped entirely (1 session dropped).
- **Behavioral trials without ephys coverage**: excluded via `obs_intervals` matching.
- **Tongue tracking gaps**: bins with no camera frame use fallback to the last available frame before the bin, rather than marking as missing/not visible.

`free_water` trials are NOT filtered out, unlike the reference. Tongue likelihood is NOT used for filtering, unlike the reference.

ii.
```python
if n_good_units == 0:
    return None
```

```python
fallback_missing = int(np.sum(right_idx <= left_idx))
sample_idx = np.clip(right_idx - 1, 0, len(timestamps) - 1)
```

iii. The CONVERSION_NOTES.md states that 149,144 bins required previous-frame fallback for tongue binning, and that 2,452 "all neural data is zero" warnings were reported by the verification script.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are reading each NWB file via h5py and the spike binning loop. The spike binning function iterates over all units, performing searchsorted and add.at operations per unit.

ii.
```python
for unit_idx, unit_end in enumerate(spike_times_index):
    ...
    np.add.at(counts[:, :, good_unit_col], (valid_trial_idx, bin_idx), 1)
```

iii. No specific timing information is provided in the CONVERSION_NOTES.md beyond noting that full conversion completes successfully.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops could potentially be vectorized:
1. The per-unit spike binning loop in `bin_spike_counts_for_good_units` iterates over each unit individually.
2. The per-trial photostim loop in `bin_photostim_series` iterates over each trial.

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

iii. The per-unit loop is inherent to the ragged spike time storage. The photostim loop could be vectorized (the reference does this). The choice computation also uses a per-trial Python loop.

## 10-c. What processing does the code repeat multiple times?

i. The `decode_array` and `decode_scalar` functions are called repeatedly for byte-to-string conversion. The photostim `as_float_or_none` is called per trial per field. Otherwise, no major repeated computation.

ii.
```python
def decode_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return value
```

iii. N/A

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes detailed statistics counters (`choice_sources`, `outcome_counts`, `early_counts`, `tongue_bin_fallback_count`) that are stored in metadata but not used by the decoder. The `choice` computation using actual lick times with hierarchical fallback is more complex than needed; the simpler instruction x outcome derivation would suffice. The AI also computes and stores both full and sample datasets in a single run.

ii.
```python
source_counter["post_go_lick"] += 1
...
source_counter["any_trial_lick"] += 1
...
source_counter["instruction_fallback"] += 1
```

iii. The statistics are useful for debugging but represent extra computation.
