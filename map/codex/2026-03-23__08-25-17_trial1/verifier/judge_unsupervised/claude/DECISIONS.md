# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB/HDF5 files stored in a DANDI-style directory structure (`data/sub-<subject_id>/<session>.nwb`). It discovers all session files using a glob pattern `sub-*/*.nwb`, then processes each file sequentially using `h5py` for direct HDF5 access (not PyNWB). Each NWB file contains one session's data including units (spike times, quality metrics, brain region annotations), behavioral events (go cue times, lick times, sample/delay start times), trial table (trial_instruction, early_lick, outcome, photostim fields), and behavioral time series (tongue tracking).

ii.
```python
def get_session_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))

# In process_session():
with h5py.File(file_path, "r") as h5:
    subject_id, session_id = get_session_identity(file_path)
    units = h5["units"]
    trials = h5["intervals"]["trials"]
    # ... loads spike_times, trial table fields, behavioral events, tongue tracking
```

iii. The AI chose `h5py` over PyNWB for performance, noting that "Full PyNWB object loading produces avoidable overhead and warning spam" (CONVERSION_NOTES Step 6). The glob-based discovery ensures all NWB files across all subject directories are found.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the parent directory name of each NWB file (`sub-<subject_id>`). The `subject_id` is extracted by stripping the `sub-` prefix. After processing all sessions, unique subjects are collected and sorted, and a `subject_idx` array maps each session to its subject.

ii.
```python
def get_session_identity(file_path: Path) -> tuple[str, str]:
    subject_id = file_path.parent.name.replace("sub-", "")
    session_id = file_path.stem
    return subject_id, session_id

# In build_dataset():
subjects = sorted({r.subject_id for r in results})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI noted 28 subjects across 174 NWB files, consistent with the paper's report of 28 mice (CONVERSION_NOTES Steps 2-3).

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are processed individually. Sessions with zero `classification == "good"` units are skipped (one session excluded, yielding 173 analyzable sessions). In `--sample` mode, files are limited to the first 12 NWB files, stopping after 2 successful sessions.

ii.
```python
# In process_session():
classification = np.char.lower(decode_str_array(units["classification"][:]))
good_unit_idx = np.flatnonzero(classification == "good")
if len(good_unit_idx) == 0:
    print(f"Skipping {session_id}: zero good units")
    return None
```

iii. The AI documented that excluding the one zero-good-unit session yields 173 sessions, matching the paper's "173 behavioral sessions" (CONVERSION_NOTES Step 4).

## 1-d. How are the data split into trials?

i. Trials are initially identified from the behavioral trial table (`intervals/trials`). Go cue times are loaded from `BehavioralEvents/go_start_times`. Trials are then filtered to keep only those whose full neural extraction window `[-2.5s, +1.5s]` relative to go cue falls within the session's observation interval (union of good-unit `obs_intervals`).

ii.
```python
def select_trial_indices(
    go_times_all, good_unit_obs_intervals,
) -> tuple[np.ndarray, float, float]:
    session_obs_start = float(np.min(good_unit_obs_intervals[:, 0]))
    session_obs_stop = float(np.max(good_unit_obs_intervals[:, 1]))
    full_window_mask = (
        (go_times_all + WINDOW_START_S >= session_obs_start)
        & (go_times_all + WINDOW_END_S <= session_obs_stop)
    )
    trial_idx = np.flatnonzero(full_window_mask)
    return trial_idx, session_obs_start, session_obs_stop
```

iii. The AI initially assumed the first N behavioral trials aligned with `is_good_trials` columns, but discovered through sanity checks that many sessions have offset trial blocks. This was documented as a bug fix in Step 10, with the obs_intervals-based selection resolving the issue.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies two levels of trial filtering: (1) trials must have their full `[-2.5, 1.5]s` go-aligned window within the session observation interval; (2) after neural binning, trials where all neural activity is zero are dropped. The AI does NOT apply the reference code's `get_regular_trial_mask` which excludes early-lick, auto-water, free-water, no-response, and photostimulation trials. The justification is that early lick, outcome (which includes ignore/no-response), and photostimulation are decoder inputs/outputs.

ii.
```python
# After neural binning:
nonzero_trial_mask = np.any(neural_session > 0, axis=(1, 2))
dropped_zero_trials = int(np.sum(~nonzero_trial_mask))
if dropped_zero_trials:
    neural_session = neural_session[nonzero_trial_mask]
    # ... filters all arrays consistently
```

iii. The AI explicitly justified: "Retain early-lick, miss, hit, ignore, and photostimulation trials because these are either decoder outputs or decoder inputs. Do not apply the reference code's `regular_trial_mask` wholesale." (CONVERSION_NOTES Step 5, Key Decision 8). This is consistent with the instructions which specify early_lick, outcome, and photostim as decoder variables.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (ragged array of spike timestamps) for units with `units/classification == "good"`. The ragged array is indexed via `units/spike_times_index`.

ii.
```python
spike_times = units["spike_times"][:].astype(np.float64)
spike_times_index = units["spike_times_index"][:]
spike_starts, spike_ends = get_ragged_row_bounds(spike_times_index)
```

iii. The AI identified that the NWB files contain raw spike times from Kilosort2, with quality classification from a logistic-regression classifier (CONVERSION_NOTES Steps 1-2).

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins over the `[-2.5, 1.5]s` window relative to go cue. Spike counts are converted to firing rates (Hz) by dividing by bin width (0.05s). The result is stored as float16.

ii.
```python
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5

# In process_session():
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
flat_edges = abs_edges.reshape(-1)
for unit_pos, unit_idx in enumerate(good_unit_idx):
    spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
    edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
    counts = np.diff(edge_indices, axis=1).astype(np.float32)
    fr = counts / bin_width
```

iii. The AI noted this deliberately departs from the reference code's 40ms bin width / 3.4ms stride, as the task instructions explicitly require 50ms bins (CONVERSION_NOTES Step 5, Key Decision 4).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == "good"` are included. Per-unit, per-trial validity is checked using `is_good_trials` when the column count matches the selected trial count; otherwise, validity is determined from `obs_intervals`. Invalid unit-trial combinations have their firing rates set to zero.

ii.
```python
classification = np.char.lower(decode_str_array(units["classification"][:]))
good_unit_idx = np.flatnonzero(classification == "good")

# Per-unit validity:
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

iii. The AI's neuron QC uses the NWB `classification` field which stores classifier-based good unit labels, matching the reference code's use of classifier-selected good units from QC files (CONVERSION_NOTES Step 1).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. The go cue timestamps are loaded from `acquisition/BehavioralEvents/go_start_times/timestamps`. Bin edges are computed as offsets from the go cue time: `go_time + [-2.5, -2.45, ..., 1.45, 1.5]`.

ii.
```python
go_times_all = h5["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][:n_behavior_trials].astype(np.float64)
# ...
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
```

iii. This matches the instructions ("Temporally align based on Go cue onset") and the reference code which treats spike times as go-cue aligned (CONVERSION_NOTES Step 1: "in Susu's data, spike_times are relative to go cue time").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50ms (non-overlapping bins). This produces 80 time bins over the [-2.5, 1.5]s window. Bin edges are constructed with `np.arange(WINDOW_START_S, WINDOW_END_S + BIN_WIDTH_S * 0.5, BIN_WIDTH_S)`. No temporal rebinning is applied; spikes are directly binned at 50ms resolution from raw spike times.

ii.
```python
BIN_WIDTH_S = 0.05

def bin_edges_and_centers():
    edges = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_WIDTH_S * 0.5, BIN_WIDTH_S, dtype=np.float64)
    centers = edges[:-1] + BIN_WIDTH_S / 2.0
    return edges, centers
```

iii. The 50ms bin width is explicitly required by the instructions ("Use 50-ms-width bins for computing firing rates"). The reference code uses 40ms bin width with 3.4ms stride, which the AI correctly identified as needing to change per the task spec.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This input is derived from `acquisition/BehavioralEvents/sample_start_times/timestamps` (the tone/sample onset event times) and `go_start_times/timestamps` (the go cue times). The sample onset is identified as the earliest `sample_start_times` event within each trial's time window.

ii.
```python
sample_start_times = h5["acquisition"]["BehavioralEvents"]["sample_start_times"]["timestamps"][:].astype(np.float64)
# ...
sample_slice_starts, sample_slice_ends = event_slices_for_trials(sample_start_times, trial_start, trial_stop)
for trial in range(n_trials):
    events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]
    if len(events):
        sample_onset_abs[trial] = events[0]
    else:
        sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO
```

iii. The AI noted that the expected sample onset relative to go cue is approximately -1.85s, and uses a fallback of `go_time - 1.85` when no sample event is found within the trial window (CONVERSION_NOTES Step 5).

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the sample onset time is found relative to the go cue. Then for each time bin, the value is computed as `bin_center_relative_to_go - sample_onset_relative_to_go`, giving a continuous, monotonically increasing time series that represents seconds elapsed since tone onset. A fallback of -1.85s relative to go cue is used when no sample_start event is found.

ii.
```python
sample_onset_rel_go = sample_onset_abs - go_times
# ...
inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
```

iii. This produces a continuous time-varying input as specified by the instructions ("Time from tone onset in seconds (continuous, time-varying)").

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both the neural data and this input use the same bin centers relative to go cue onset. The input is computed at the same 80 time bin centers as the neural data, so they are inherently aligned.

ii.
```python
inp = np.zeros((2, n_bins), dtype=np.float32)
inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
```

iii. Alignment is guaranteed by using the same `bin_centers_rel` array for both neural binning and input construction.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation input is derived from the trial table fields `photostim_onset` and `photostim_duration` in `intervals/trials`. Non-stimulation trials have the value `"N/A"` for these fields.

ii.
```python
photostim_onset_all = decode_str_array(trials["photostim_onset"][:])
photostim_duration_all = decode_str_array(trials["photostim_duration"][:])
```

iii. The AI verified that trial-table photostim onset/duration matched the behavioral-event photostim timestamps in spot checks (CONVERSION_NOTES Step 7).

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial with photostimulation, the onset time is converted to relative-to-go-cue timing: `trial_start + photostim_onset - go_time`. The photostim is represented as a binary time series where bins within `[stim_on, stim_on + duration)` are set to 1, and all other bins are 0.

ii.
```python
if str(photostim_onset[trial]) != "N/A":
    stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
    stim_rel_off = stim_rel_on + float(photostim_duration[trial])
    inp[1] = ((bin_centers_rel >= stim_rel_on) & (bin_centers_rel < stim_rel_off)).astype(np.float32)
```

iii. The AI justified using a time-varying binary representation because "the decoder task requests the on/off state at every time point" (CONVERSION_NOTES Step 5, Key Decision 6).

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostimulation binary vector uses the same `bin_centers_rel` as the neural data, so alignment is inherent. The photostim onset timing is converted from trial-relative to go-cue-relative coordinates to match the neural alignment.

ii.
```python
# Same bin_centers_rel used for both neural and input:
inp[1] = ((bin_centers_rel >= stim_rel_on) & (bin_centers_rel < stim_rel_off)).astype(np.float32)
```

iii. The reference code also aligns stimulation times to go cue (subtracting `gocue_time` from `stimulation` columns in `process_one_sess`).

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from `acquisition/BehavioralEvents/left_lick_times/timestamps` and `right_lick_times/timestamps`. As a fallback, `intervals/trials/trial_instruction` is used for no-lick trials.

ii.
```python
left_lick_times = h5["acquisition"]["BehavioralEvents"]["left_lick_times"]["timestamps"][:].astype(np.float64)
right_lick_times = h5["acquisition"]["BehavioralEvents"]["right_lick_times"]["timestamps"][:].astype(np.float64)
```

iii. The AI noted that the reference code stores `lick_directions` per trial, while the NWB format requires inferring choice from raw lick event timestamps (CONVERSION_NOTES Step 5).

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. A three-tier fallback hierarchy is used: (1) first post-go-cue lick determines direction (left=0, right=1); (2) if no post-go lick, first lick anywhere in the trial is used; (3) if no lick at all, trial instruction is used as fallback. The result is a per-trial scalar broadcast to all time bins.

ii.
```python
def infer_choice_for_trial(...) -> tuple[int, str]:
    # First try post-go licks
    left_post = left_lick_times[left_go:left_stop]
    right_post = right_lick_times[right_go:right_stop]
    if len(left_post) or len(right_post):
        first_left = left_post[0] if len(left_post) else np.inf
        first_right = right_post[0] if len(right_post) else np.inf
        return (0, "post_go_lick") if first_left < first_right else (1, "post_go_lick")
    # Then any lick in trial
    # ...
    # Fallback to instruction
    return (0 if instruction == "left" else 1, "instruction_fallback")
```

iii. The AI documented the fallback as necessary because "ignore trials have no post-go lick but the decoder spec still requires a binary choice output" (CONVERSION_NOTES Step 5, Key Decision 7).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `intervals/trials/outcome` which contains string values `"hit"`, `"miss"`, and `"ignore"`.

ii.
```python
outcome_all = np.char.lower(decode_str_array(trials["outcome"][:]))
```

iii. The AI confirmed outcome values in the NWB match expected categories from the papers.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The string outcome values are mapped to integers: `ignore -> 0`, `miss -> 1`, `hit -> 2`. The result is a per-trial scalar broadcast to all time bins.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome_trials = np.array([outcome_map[str(x)] for x in outcome], dtype=np.int8)
# ...
out[1] = outcome_trials[trial]  # broadcast to all bins
```

iii. This matches the instruction spec: "Outcome (ignore = 0, miss = 1, hit = 2, per-trial)".

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from `intervals/trials/early_lick` which contains string values `"early"` and `"no early"`.

ii.
```python
early_lick_all = np.char.lower(decode_str_array(trials["early_lick"][:]))
```

iii. The AI confirmed these values in sample NWB file inspection (CONVERSION_NOTES Step 2).

## 7-b. What processing is involved in computing `output` *Early lick*?

i. String values are mapped to integers: `"no early" -> 0`, `"early" -> 1`. The result is per-trial, broadcast to all time bins.

ii.
```python
early_map = {"no early": 0, "early": 1}
early_trials = np.array([early_map[str(x)] for x in early_lick], dtype=np.int8)
# ...
out[2] = early_trials[trial]
```

iii. This matches the instruction spec: "Early lick (no = 0, yes = 1, per-trial)".

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains a `(n_frames, 3)` array with columns `(tongue_x, tongue_y, tongue_likelihood)` and associated `timestamps`.

ii.
```python
tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_values = tongue_group["data"][:].astype(np.float64)
tongue_y = tongue_values[:, 1]
tongue_likelihood = tongue_values[:, 2]
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
```

iii. The AI identified this as the side-camera tongue tracking data sampled at ~300 Hz (CONVERSION_NOTES Step 2).

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Tongue y values are aligned to go cue and binned into 50ms bins. Within each bin, the last available tongue_y sample is taken (matching the reference code's `align_markers_between_lims` convention). NaN values are used for bins with no tongue data.

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

iii. The AI noted: "Reference marker alignment uses the last frame within each time step rather than averaging" (CONVERSION_NOTES Step 5).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session percentiles are computed over all valid (non-NaN) binned tongue y values. Values below the 40th percentile are category 0, between 40th and 60th percentile are category 1, and above 60th percentile are category 2.

ii.
```python
valid_values = tongue_y_binned[np.isfinite(tongue_y_binned)]
tongue_p40 = float(np.percentile(valid_values, 40))
tongue_p60 = float(np.percentile(valid_values, 60))
tongue_discrete = np.zeros_like(tongue_y_binned, dtype=np.int8)
tongue_discrete[(tongue_y_binned >= tongue_p40) & (tongue_y_binned <= tongue_p60)] = 1
tongue_discrete[tongue_y_binned > tongue_p60] = 2
```

iii. This matches the instructions: "0: < 40th percentile, 1: 40th to 60th percentile, 2: > 60th percentile of y-position over the session". Note the boundary handling: values exactly at the 40th percentile get category 1, while values exactly at the 60th percentile also get category 1 (using `<=`).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y is aligned using the same go-cue-relative bin edges as the neural data. The absolute bin edges are computed as `go_time + bin_edges_rel`, ensuring tongue data and neural data share the same temporal grid.

ii.
```python
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
```

iii. The AI confirmed alignment through processing plots showing raw tongue data overlaid with binned values aligned to go cue.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled: (1) Missing sample onset events: fallback to `go_time - 1.85s`; (2) No post-go licks for choice: fallback hierarchy to any trial lick, then instruction; (3) Tongue y bins with no data: filled with NaN, then set to category 0 in discretization; (4) Units with `is_good_trials` column count mismatch: falls back to obs_intervals; (5) Sessions with zero good units: skipped entirely; (6) Trials with all-zero neural activity: dropped after binning.

ii.
```python
# Sample onset fallback:
sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO

# is_good_trials fallback:
uses_direct_is_good_trials = is_good_trials_raw.shape[1] == len(selected_trial_idx)
if uses_direct_is_good_trials:
    is_good_trials = is_good_trials_raw.copy()
else:
    is_good_trials = np.ones((len(good_unit_idx), len(selected_trial_idx)), dtype=bool)
```

iii. The AI documented discovery of the trial-offset bug as a major edge case: "86 kept sessions with a nonzero behavioral-trial start offset" and "120 kept sessions where the number of trials with full neural-window support did not equal is_good_trials.shape[1]" (CONVERSION_NOTES Step 10).

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the neural spike binning loop, which iterates over all good units in a session and performs `np.searchsorted` on the full spike train for each unit across all trial/bin edges. Loading the full NWB file (all spike times, tongue timestamps) is also significant. The AI reported ~1.6s per session for sample conversion, estimating ~4.6 minutes for all 173 sessions.

ii.
```python
for unit_pos, unit_idx in enumerate(good_unit_idx):
    spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
    edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
    counts = np.diff(edge_indices, axis=1).astype(np.float32)
    fr = counts / bin_width
```

iii. The AI documented run time estimates in CONVERSION_NOTES Step 7.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit spike binning loop (iterating over `good_unit_idx`) could potentially be vectorized further, though the `searchsorted` approach is already efficient. The per-trial loops for constructing inputs, choice inference, and sample onset finding are Python-level loops that could be vectorized. The tongue y binning is already vectorized.

ii.
```python
# Per-trial loops that could be vectorized:
for trial in range(n_trials):
    events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]
    # ...

for trial in range(n_trials):
    choice_val, source = infer_choice_for_trial(...)
```

iii. The AI did vectorize the major bottleneck (spike binning) but left smaller loops as Python-level iterations.

## 10-c. What processing does the code repeat multiple times?

i. The code loads all spike times from the NWB file at once, which is efficient. However, the `event_slices_for_trials` function is called separately for `sample_start_times` and `delay_start_times`, performing similar searchsorted operations. The output construction loop repeats the per-trial broadcast pattern 4 times. The `bin_edges_and_centers()` function is called once per session but computes the same values each time.

ii.
```python
sample_slice_starts, sample_slice_ends = event_slices_for_trials(sample_start_times, trial_start, trial_stop)
delay_slice_starts, delay_slice_ends = event_slices_for_trials(delay_start_times, trial_start, trial_stop)
```

iii. These are minor inefficiencies; the main processing is not significantly repeated.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes delay onset times (`delay_start_times`, `delay_slice_starts/ends`) which are only used in the processing plots but not in the actual converted data. The `tongue_likelihood` values are loaded but only used for QC diagnostics in plots. The `is_good_trials` per-unit validity is computed and applied (setting invalid unit-trial firing rates to zero), but this zeroing may not be the ideal approach for downstream decoders. The sample onset fallback counter and choice source counts are tracked for statistics but don't affect the output.

ii.
```python
# Delay times loaded but only used in plots:
delay_start_times = h5["acquisition"]["BehavioralEvents"]["delay_start_times"]["timestamps"][:].astype(np.float64)
delay_slice_starts, delay_slice_ends = event_slices_for_trials(delay_start_times, trial_start, trial_stop)

# Tongue likelihood loaded but only used in plot payload:
tongue_likelihood = tongue_values[:, 2]
```

iii. The delay time processing and tongue likelihood loading are unnecessary for the core conversion but add minimal overhead.
