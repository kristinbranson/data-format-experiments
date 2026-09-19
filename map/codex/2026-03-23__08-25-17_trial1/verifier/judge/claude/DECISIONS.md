# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all NWB files via a sorted glob over `data/sub-*/*.nwb` (174 files). Each file is opened with `h5py` (not `pynwb`) and processed sequentially. Trials, units, behavioral events, and tongue tracking are read from HDF5 groups within each file.

ii.
```python
def get_session_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))
```
```python
with h5py.File(file_path, "r") as h5:
    units = h5["units"]
    trials = h5["intervals"]["trials"]
    ...
    go_times_all = h5["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][:n_behavior_trials].astype(np.float64)
```

iii. The AI chose `h5py` over `pynwb` for performance, noting it avoids "heavy object construction and warning spam." The glob pattern and sort order are the same as the reference approach. From CONVERSION_NOTES.md: "Used h5py directly instead of PyNWB for the conversion path."

## 1-b. How are the data split into subjects?

i. Subject ID is parsed from the folder name by stripping the `sub-` prefix (e.g., `sub-440956` becomes `440956`). Subjects are deduplicated, sorted, and indexed.

ii.
```python
def get_session_identity(file_path: Path) -> tuple[str, str]:
    subject_id = file_path.parent.name.replace("sub-", "")
    session_id = file_path.stem
    return subject_id, session_id
```
```python
subjects = sorted({r.subject_id for r in results})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. From CONVERSION_NOTES.md Step 5: "Subject order follows converted session order." The folder-name-derived IDs produce the same numeric strings as the NWB `subject.subject_id` field.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. Session ID is the file stem (e.g., `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`). Session order follows the sorted file list.

ii.
```python
all_files = get_session_files(data_dir)
```
```python
subject_id, session_id = get_session_identity(file_path)
```

iii. The AI recognized that the DANDI dataset stores one session per file, so no further grouping is needed. 173 sessions are retained after excluding the one session with zero good units.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`h5["intervals"]["trials"]`). Go-cue timestamps are loaded from `BehavioralEvents/go_start_times`. The AI also uses `units/is_good_trials` to check which trials have valid neural data.

ii.
```python
trials = h5["intervals"]["trials"]
n_behavior_trials = int(len(trials["id"]))
trial_start_all = trials["start_time"][:].astype(np.float64)
...
go_times_all = h5["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][:n_behavior_trials].astype(np.float64)
```

iii. From CONVERSION_NOTES.md Step 9: the AI notes the trial table is the canonical source of trial definitions.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials using the union of good-unit observation intervals: it computes the minimum obs_start and maximum obs_stop across all good units, then keeps only trials whose full [-2.5, +1.5] s go-aligned window falls within that range. Additionally, after neural binning, trials with all-zero neural activity are dropped. No behavioral quality filter (free_water, early_lick, etc.) is applied at the trial-selection stage.

ii.
```python
def select_trial_indices(go_times_all, good_unit_obs_intervals):
    session_obs_start = float(np.min(good_unit_obs_intervals[:, 0]))
    session_obs_stop = float(np.max(good_unit_obs_intervals[:, 1]))
    full_window_mask = (
        (go_times_all + WINDOW_START_S >= session_obs_start)
        & (go_times_all + WINDOW_END_S <= session_obs_stop)
    )
    trial_idx = np.flatnonzero(full_window_mask)
    ...
```
```python
nonzero_trial_mask = np.any(neural_session > 0, axis=(1, 2))
dropped_zero_trials = int(np.sum(~nonzero_trial_mask))
if dropped_zero_trials:
    neural_session = neural_session[nonzero_trial_mask]
    ...
```

iii. From CONVERSION_NOTES.md Step 5: "Do require that the full neural decoding window is actually supported by the raw session observation interval." The AI chose not to filter free_water trials explicitly, relying instead on the obs_intervals window and all-zero-neural detection to exclude them implicitly. This results in 51,346 total trials, far fewer than the reference's ~90,860.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (ragged spike time arrays) for units with `classification == 'good'`. Go-cue timestamps from `BehavioralEvents/go_start_times` define the bin edges.

ii.
```python
spike_times = units["spike_times"][:].astype(np.float64)
spike_times_index = units["spike_times_index"][:]
spike_starts, spike_ends = get_ragged_row_bounds(spike_times_index)
```

iii. The AI uses the same source variable (`spike_times`) as the reference.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 50 ms non-overlapping bins aligned to the go cue. Spike counts are divided by the bin width (0.05 s) to get firing rates in Hz. The rates are stored as `float16`. The AI also uses `is_good_trials` to zero out firing rates for unit-trial pairs flagged as invalid.

ii.
```python
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
flat_edges = abs_edges.reshape(-1)
for unit_pos, unit_idx in enumerate(good_unit_idx):
    spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
    edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
    counts = np.diff(edge_indices, axis=1).astype(np.float32)
    fr = counts / bin_width
    ...
    neural_session[:, unit_pos, :] = fr.astype(np.float16)
```

iii. The AI noted using `float16` to "reduce pickle size while remaining valid floating-point input for the decoder." The core binning logic (searchsorted + diff) is the same approach as the reference.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. Sessions with zero good units are dropped (1 session). Additionally, the AI uses a per-unit `is_good_trials` matrix (when available) to zero out firing rates for invalid unit-trial pairs, and a per-unit `obs_intervals` fallback.

ii.
```python
classification = np.char.lower(decode_str_array(units["classification"][:]))
good_unit_idx = np.flatnonzero(classification == "good")
if len(good_unit_idx) == 0:
    return None
```
```python
if uses_direct_is_good_trials:
    invalid_trials = ~is_good_trials[unit_pos]
else:
    obs_start = unit_obs_intervals[unit_pos, 0]
    obs_stop = unit_obs_intervals[unit_pos, 1]
    valid_obs = (go_times + WINDOW_START_S >= obs_start) & (go_times + WINDOW_END_S <= obs_stop)
    invalid_trials = ~valid_obs
if np.any(invalid_trials):
    fr[invalid_trials] = 0.0
```

iii. From CONVERSION_NOTES.md: "Use units with classification == good as the primary QC filter." The `is_good_trials` zeroing is an additional per-unit-per-trial quality step not present in the reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue. Absolute bin edges are computed as `go_times[:, None] + bin_edges_rel[None, :]`, and spikes are binned against these absolute edges.

ii.
```python
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
flat_edges = abs_edges.reshape(-1)
```

iii. Same approach as the reference: bin edges relative to go cue added to each trial's absolute go-cue time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins spanning -2.5 s to +1.5 s relative to the go cue. No rebinning is applied (spikes are binned directly from raw spike times).

ii.
```python
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
```
```python
edges = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_WIDTH_S * 0.5, BIN_WIDTH_S, dtype=np.float64)
centers = edges[:-1] + BIN_WIDTH_S / 2.0
```

iii. Matches the instructions and the reference.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times` (tone onset events) and `go_start_times` (go cue). For each trial, the AI uses the **first** (earliest) `sample_start_times` event that falls within the trial's [start_time, stop_time] window.

ii.
```python
sample_start_times = h5["acquisition"]["BehavioralEvents"]["sample_start_times"]["timestamps"][:].astype(np.float64)
...
sample_slice_starts, sample_slice_ends = event_slices_for_trials(sample_start_times, trial_start, trial_stop)
for trial in range(n_trials):
    events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]
    if len(events):
        sample_onset_abs[trial] = events[0]
    else:
        sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO
```

iii. From CONVERSION_NOTES.md Step 5: "Use the earliest sample-start event in each trial as tone onset. This preserves replay-induced timing shifts visible in early-lick trials." The AI uses a fallback of `go_time - 1.85s` when no sample event is found within the trial window.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each bin, the value is `bin_center_relative_to_go - sample_onset_relative_to_go`, giving seconds since tone onset.

ii.
```python
sample_onset_rel_go = sample_onset_abs - go_times
...
inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
```

iii. Algebraically equivalent to the reference's `CENTERS + (go - tone)`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both neural data and the time-from-tone input share the same bin centers relative to the go cue, ensuring alignment.

ii.
```python
inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
```

iii. Same alignment approach as the reference.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, with `start_time` and go_time used to compute relative timing.

ii.
```python
photostim_onset_all = decode_str_array(trials["photostim_onset"][:])
photostim_duration_all = decode_str_array(trials["photostim_duration"][:])
```

iii. Same source variables as the reference.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series: 1 where the bin center falls within `[onset, onset+duration)` relative to go cue, 0 otherwise. Non-stimulated trials (marked `"N/A"`) remain all-zero.

ii.
```python
if str(photostim_onset[trial]) != "N/A":
    stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
    stim_rel_off = stim_rel_on + float(photostim_duration[trial])
    inp[1] = ((bin_centers_rel >= stim_rel_on) & (bin_centers_rel < stim_rel_off)).astype(np.float32)
```

iii. Same approach as the reference.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The onset/offset are expressed relative to the go cue, matching the neural bin centers.

ii. Same code as 4-b above.

iii. Same alignment approach as the reference.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from the actual lick event timestamps: `left_lick_times` and `right_lick_times` from `BehavioralEvents`. It uses a three-level fallback: (1) first post-go lick, (2) any lick in the trial, (3) trial instruction for no-lick ignore trials. Choice is always left (0) or right (1) with no separate "no lick" class.

ii.
```python
left_lick_times = h5["acquisition"]["BehavioralEvents"]["left_lick_times"]["timestamps"][:].astype(np.float64)
right_lick_times = h5["acquisition"]["BehavioralEvents"]["right_lick_times"]["timestamps"][:].astype(np.float64)
...
def infer_choice_for_trial(trial_start, trial_stop, go_time, instruction, left_lick_times, right_lick_times):
    ...
    left_post = left_lick_times[left_go:left_stop]
    right_post = right_lick_times[right_go:right_stop]
    if len(left_post) or len(right_post):
        first_left = left_post[0] if len(left_post) else np.inf
        first_right = right_post[0] if len(right_post) else np.inf
        return (0, "post_go_lick") if first_left < first_right else (1, "post_go_lick")
    ...
    return (0 if instruction == "left" else 1, "instruction_fallback")
```

iii. From CONVERSION_NOTES.md Step 5: "For hit/miss trials, decode actual lick direction from lick-event timing. For ignore trials with no lick, use the fallback hierarchy." The AI explicitly chose to infer choice from raw lick events rather than deriving it algebraically from `trial_instruction` x `outcome`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0 (left) or 1 (right) with only 2 classes. No "no lick" class is defined. Ignore trials get a choice value assigned via the instruction fallback. The value is repeated across all 80 time bins.

ii.
```python
choice_trials[trial] = choice_val
...
out[0] = choice_trials[trial]
```
```python
"output_values": [
    ["left", "right"],
    ...
]
```

iii. The AI's `output_values` for choice has only 2 values (`["left", "right"]`), missing the "no lick" class specified in the instructions.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table (`"ignore"`, `"miss"`, `"hit"`).

ii.
```python
outcome_all = np.char.lower(decode_str_array(trials["outcome"][:]))
...
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome_trials = np.array([outcome_map[str(x)] for x in outcome], dtype=np.int8)
```

iii. Same source and approach as the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to ignore=0, miss=1, hit=2. Repeated across all 80 bins.

ii.
```python
out[1] = outcome_trials[trial]
```

iii. Same as the reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table (`"no early"`, `"early"`).

ii.
```python
early_lick_all = np.char.lower(decode_str_array(trials["early_lick"][:]))
...
early_map = {"no early": 0, "early": 1}
early_trials = np.array([early_map[str(x)] for x in early_lick], dtype=np.int8)
```

iii. Same source and approach as the reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to no=0, yes=1. Repeated across all 80 bins.

ii.
```python
out[2] = early_trials[trial]
```

iii. Same as the reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, using column 1 (`tongue_y`) and associated timestamps.

ii.
```python
tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_values = tongue_group["data"][:].astype(np.float64)
tongue_y = tongue_values[:, 1]
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
```

iii. Same source variable as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI takes the **last** frame in each 50 ms bin (not the mean). No likelihood-based filtering is applied (all frames are used regardless of tongue visibility). Percentiles (40th and 60th) are computed over all finite binned values across the session and used to discretize into 3 classes (0, 1, 2). There is no separate "not visible" class.

ii.
```python
def bin_tongue_y(tongue_timestamps, tongue_y, go_times, bin_edges_rel):
    abs_edges = go_times[:, None] + bin_edges_rel[None, :]
    start_idx = np.searchsorted(tongue_timestamps, abs_edges[:, :-1], side="left")
    end_idx = np.searchsorted(tongue_timestamps, abs_edges[:, 1:], side="left") - 1
    clipped_end = np.clip(end_idx, 0, len(tongue_y) - 1)
    valid = end_idx >= start_idx
    binned = np.full(end_idx.shape, np.nan, dtype=np.float32)
    binned[valid] = tongue_y[clipped_end[valid]].astype(np.float32)
    return binned, valid
```
```python
valid_values = tongue_y_binned[np.isfinite(tongue_y_binned)]
tongue_p40 = float(np.percentile(valid_values, 40))
tongue_p60 = float(np.percentile(valid_values, 60))
tongue_discrete = np.zeros_like(tongue_y_binned, dtype=np.int8)
tongue_discrete[(tongue_y_binned >= tongue_p40) & (tongue_y_binned <= tongue_p60)] = 1
tongue_discrete[tongue_y_binned > tongue_p60] = 2
```

iii. From CONVERSION_NOTES.md Step 5: "Use tongue_likelihood only for QC diagnostics, not thresholding, to stay close to reference code." The AI chose to follow the reference code's `align_markers_between_lims` approach (last frame in bin) rather than the reference solution's mean-based approach with likelihood filtering.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Values below the 40th percentile get class 0, values between 40th and 60th percentile get class 1, values above 60th percentile get class 2. NaN bins (where no frames exist) receive class 0 by default (initialized to zero and not modified). There is no class 3 "not visible."

ii.
```python
tongue_discrete = np.zeros_like(tongue_y_binned, dtype=np.int8)
tongue_discrete[(tongue_y_binned >= tongue_p40) & (tongue_y_binned <= tongue_p60)] = 1
tongue_discrete[tongue_y_binned > tongue_p60] = 2
```
```python
"output_values": [
    ...
    ["lt_p40", "p40_to_p60", "gt_p60"],
]
```

iii. The instructions specify 4 categories: 0 (< 40th), 1 (40th-60th), 2 (> 60th), 3 (not visible). The AI only implements 3 categories, omitting the "not visible" class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are aligned to the go cue using the same bin edges as the neural data. Searchsorted on the camera timestamps at the absolute bin edges finds the relevant frames for each bin.

ii.
```python
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
start_idx = np.searchsorted(tongue_timestamps, abs_edges[:, :-1], side="left")
end_idx = np.searchsorted(tongue_timestamps, abs_edges[:, 1:], side="left") - 1
```

iii. Same alignment approach as the reference: camera timestamps share the session-absolute clock with spikes and events.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases are handled:
- **Zero good units**: sessions are skipped entirely (1 session dropped).
- **Invalid unit-trial pairs**: `is_good_trials` matrix zeros out firing rates for specific unit-trial combinations; all-zero neural trials are then dropped entirely.
- **Missing tongue frames**: bins without frames receive NaN in the binned array, but are then silently assigned class 0 in the discrete output (a bug, as NaN comparisons are all False and the default is 0).
- **Missing sample onset**: falls back to `go_time - 1.85s` as a default.

ii.
```python
if len(good_unit_idx) == 0:
    return None
```
```python
if np.any(invalid_trials):
    fr[invalid_trials] = 0.0
```
```python
sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO
```

iii. The AI documented handling for zero-good-unit sessions and used `is_good_trials` as an additional quality layer. The sample onset fallback is not present in the reference.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file via h5py dominates. Full conversion takes ~266 s for 174 files (~1.54 s per session). The per-unit searchsorted spike binning and tongue tracking array reads are the main within-session costs. Pickling the output takes additional time.

ii. N/A

iii. From CONVERSION_NOTES.md Step 7: "~1.60 s / kept session" estimated; actual was ~1.54 s.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops remain: (1) per-unit spike binning loop (`for unit_pos, unit_idx in enumerate(good_unit_idx)`), (2) per-trial input/output construction loop. The per-trial tongue binning is vectorized but uses a fallback per-trial approach for the input construction.

ii.
```python
for unit_pos, unit_idx in enumerate(good_unit_idx):
    spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
    edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
```
```python
for trial in range(n_trials):
    inp = np.zeros((2, n_bins), dtype=np.float32)
    inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
```

iii. The per-trial input loop could have been vectorized (as the reference does with `time_from_tone = CENTERS[None, :] + (go - tone)[:, None]`). The per-trial choice inference loop could also have been vectorized.

## 10-c. What processing does the code repeat multiple times?

i. No significant repeated processing. Each NWB file is opened once, each quantity is computed once. The bin edges are computed once via `bin_edges_and_centers()` and reused.

ii. N/A

iii. N/A

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores detailed per-session statistics (`stats` dict), choice source counts, and sample onset fallback counts. These are used for logging/diagnostics but not included in the final pickle. The `is_good_trials` handling adds complexity for zeroing out per-unit-per-trial firing rates, which the reference does not do.

ii.
```python
stats = {
    "session_id": session_id,
    ...
    "choice_source_counts": choice_source_counts,
    "sample_onset_fallbacks": sample_onset_fallbacks,
    "dropped_all_zero_neural_trials": dropped_zero_trials,
    ...
}
```

iii. The statistics computation is relatively lightweight and serves diagnostic purposes.
