# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files found by globbing `/app/data/sub-*/*.nwb`. Each NWB file represents one session. For each file, trials are read from `intervals/trials`, neural data from `units`, behavioral events from `acquisition/BehavioralEvents`, and tongue tracking from `acquisition/BehavioralTimeSeries`. The reference code loads `.mat` files exported from DataJoint, but since the available data is in NWB format, using NWB is the correct approach.

ii.
```python
nwb_paths = sorted(data_dir.glob("sub-*/*.nwb"))
# ...
for session_number, path in enumerate(nwb_paths, start=1):
    with h5py.File(path, "r") as f:
        trials = f["intervals/trials"]
        # ...
        units = f["units"]
        be = f["acquisition/BehavioralEvents"]
        tongue_group = f["acquisition/BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
```

iii. The AI's CONVERSION_NOTES.md states: "Input files are all NWB files under `/app/data/sub-*/*.nwb`. The release contains 174 NWB files." The AI systematically processes each NWB file as a session.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by reading `general/subject/subject_id` from each NWB file. A global list of unique subjects is maintained, and each session is mapped to its subject via `subject_idx`. This matches the NWB data organization where each file contains metadata about the recording subject.

ii.
```python
subject = _decode_scalar(f["general"]["subject"]["subject_id"][()])
subject = str(subject)
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
subject_idx.append(subject_to_idx[subject])
```

iii. The CONVERSION_NOTES.md reports 28 subjects in the final dataset.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. After filtering (removing sessions with zero good units and sessions with fewer than 2 valid trials), the final dataset contains 173 sessions. This matches the reference paper's reported 173 sessions.

ii.
```python
for session_number, path in enumerate(nwb_paths, start=1):
    with h5py.File(path, "r") as f:
        # ... process one session per file
        if len(good_unit_indices) == 0:
            continue
        # ...
        if n_keep_trials < 2:
            continue
```

iii. CONVERSION_NOTES.md: "The release contains 174 NWB files. One file... has zero units with `units/classification == 'good'` and is dropped. Final converted dataset therefore contains 173 sessions."

## 1-d. How are the data split into trials?

i. Trials are read from the `intervals/trials` table in each NWB file. Each trial has associated start/stop times, behavioral labels, and go cue times. The go cue times come from `acquisition/BehavioralEvents/go_start_times/timestamps`, with a consistency check that the number of go cues matches the number of trials.

ii.
```python
trials = f["intervals/trials"]
n_trials = len(trials["id"])
trial_starts = np.asarray(trials["start_time"], dtype=np.float64)
trial_stops = np.asarray(trials["stop_time"], dtype=np.float64)
# ...
go_starts = np.asarray(be["go_start_times"]["timestamps"], dtype=np.float64)
if len(go_starts) != n_trials:
    raise RuntimeError(f"{path.name}: expected {n_trials} go cues, found {len(go_starts)}")
```

iii. The AI validates trial-go cue correspondence to ensure correct temporal alignment.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three filters: (1) exclude auto_water trials, (2) exclude free_water trials, (3) restrict to trials where all good units have valid electrophysiology (using `is_good_trials` and `obs_intervals`). After neural binning, trials with all-zero neural activity are also removed. Importantly, early lick trials, ignore/no-response trials, and photostimulation trials are **kept**, because the instructions require early lick and photostimulation as decoder outputs/inputs, and outcome includes "ignore."

The reference code's `get_regular_trial_mask` also excludes early lick trials, no-response trials (correctness == -1), and photostimulation trials. However, the task instructions explicitly require these as decoder inputs/outputs, so the AI's decision to keep them is consistent with the instructions.

ii.
```python
trial_keep = np.ones(n_trials, dtype=bool)
if exclude_auto_free:
    trial_keep &= auto_water == 0
    trial_keep &= free_water == 0
# ... obs_intervals / is_good_trials filtering ...
trial_keep &= recorded_trial_mask

# After binning, remove all-zero trials:
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
session_neural = session_neural[nonzero_trial_mask]
```

iii. CONVERSION_NOTES.md: "auto_water == 1 trials are excluded. free_water == 1 trials are excluded. Early-lick, ignore, and photostimulation trials are retained because they are required by the requested decoder outputs and inputs."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times stored in `units/spike_times` (with indexing via `units/spike_times_index`). Only units where `units/classification == "good"` are included, corresponding to the classifier-based QC described in the reference paper and the spike sorting white paper.

ii.
```python
classification = _decode_str_array(units["classification"]).reshape(-1)
good_unit_indices = np.flatnonzero(classification == "good")
# ...
for pos, unit_idx in enumerate(good_unit_indices):
    unit_spikes = _extract_unit_spikes(units, int(unit_idx))
```

iii. CONVERSION_NOTES.md: "Unit inclusion follows the NWB classifier output: keep units where `units/classification == 'good'`." This corresponds to the reference paper's classifier-based quality control.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50-ms non-overlapping bins spanning -2.5s to +1.5s relative to go cue onset (80 bins total). Spike counts are divided by bin width to obtain firing rates in Hz. The reference code uses a sliding histogram with 40ms bandwidth and 3.4ms stride, but the task instructions explicitly specify 50ms bins, which the AI correctly follows.

ii.
```python
BIN_SIZE_S = 0.05
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_SIZE_S, dtype=np.float64)
# ...
abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
# ...
spike_bins = np.searchsorted(unit_spikes, abs_edges, side="left")
spike_counts = np.diff(spike_bins, axis=1).astype(np.float32) / np.float32(BIN_SIZE_S)
```

iii. CONVERSION_NOTES.md: "Neural data: firing rates in Hz from non-overlapping spike-count bins" with "80 bins of width 50 ms."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == "good"` are included. Additionally, the AI uses `is_good_trials` to restrict to trials where all good units have valid recordings. After binning, trials with all-zero neural activity across all neurons are removed.

ii.
```python
good_unit_indices = np.flatnonzero(classification == "good")
# ...
is_good_trials = np.asarray(units["is_good_trials"][good_unit_indices]).astype(bool)
# ... complex mapping using obs_intervals ...
trial_keep &= recorded_trial_mask
# ...
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
session_neural = session_neural[nonzero_trial_mask]
```

iii. CONVERSION_NOTES.md: "Trials are further restricted to those with common good-ephys support using `units/obs_intervals` plus `units/is_good_trials`. After neural binning, two residual all-zero neural trials were removed."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go cue onset. For each trial, absolute bin edges are computed by adding the relative bin edges (-2.5s to 1.5s) to the go cue time for that trial. Spikes are then counted within each absolute bin.

ii.
```python
abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
# ...
spike_bins = np.searchsorted(unit_spikes, abs_edges, side="left")
spike_counts = np.diff(spike_bins, axis=1).astype(np.float32) / np.float32(BIN_SIZE_S)
```

iii. CONVERSION_NOTES.md: "Alignment event is go cue onset from `acquisition/BehavioralEvents/go_start_times/timestamps`."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50 ms (0.05 s), producing 80 non-overlapping bins from -2.5s to +1.5s. No temporal rebinning is applied; spikes are binned directly at this resolution. This matches the task instructions ("50-ms-width bins"). The reference code uses 40ms bins with 3.4ms stride, but the instructions override this.

ii.
```python
BIN_SIZE_S = 0.05
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_SIZE_S, dtype=np.float64)
N_BINS = len(BIN_CENTERS_REL)  # 80
```

iii. CONVERSION_NOTES.md: "80 bins of width 50 ms."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Time from tone onset is derived from `acquisition/BehavioralEvents/sample_start_times/timestamps` (the tone/sample onset timestamps) and `go_start_times/timestamps` (the go cue times). The tone onset for each trial is resolved by finding the last sample_start time before the go cue.

ii.
```python
sample_starts = np.asarray(be["sample_start_times"]["timestamps"], dtype=np.float64)
tone_onsets = _resolve_sample_onsets(trial_starts, go_starts, sample_starts)
```

iii. CONVERSION_NOTES.md: "In these NWB files, `sample_start_times` can contain replayed sample epochs after early licks, so naive one-to-one trial matching is wrong."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone onset relative to go cue is computed as `tone_onset - go_time`. Then for each time bin, `time_from_tone = bin_center - tone_rel`, giving time elapsed since tone onset at each bin center. The `_resolve_sample_onsets` function handles the complication that early lick trials cause replayed sample epochs, so there can be more sample_start entries than trials.

ii.
```python
def _resolve_sample_onsets(trial_starts, go_starts, sample_starts):
    sample_idx = np.searchsorted(sample_starts, go_starts, side="right") - 1
    # ... fallback logic for edge cases ...
    return sample_onsets

# Per trial:
tone_rel = kept_tone_onsets[local_idx] - go_starts[trial_idx]
time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
```

iii. CONVERSION_NOTES.md: "For each trial, tone onset is resolved as the last `sample_start_times` timestamp occurring before that trial's go cue."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time_from_tone is computed at each bin center (same bin centers as neural data), so it is inherently aligned with the neural time bins.

ii.
```python
time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
# BIN_CENTERS_REL are the same centers used for neural binning
input_trial = np.vstack([time_from_tone, stim_on]).astype(np.float32)
```

iii. Both neural and input data use the same `BIN_CENTERS_REL` time axis.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `intervals/trials/photostim_onset`, `intervals/trials/photostim_duration`, and `intervals/trials/photostim_power` (though power is read but not directly used for the binary on/off signal). The onset is interpreted relative to trial start time.

ii.
```python
photostim_onset = _parse_optional_float_array(trials["photostim_onset"])
photostim_duration = _parse_optional_float_array(trials["photostim_duration"])
photostim_power = _parse_optional_float_array(trials["photostim_power"])
```

iii. CONVERSION_NOTES.md: "Trial photostim timing uses `intervals/trials/photostim_onset` and `photostim_duration`."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Photostimulation is encoded as a binary time series (1 when photostim is active, 0 otherwise). The absolute stimulation start is computed as `trial_start + photostim_onset`, and the stop as `start + photostim_duration`. A bin is marked as 1 if its center falls within the stimulation interval [start, stop).

ii.
```python
stim_on = np.zeros(N_BINS, dtype=np.float32)
if np.isfinite(kept_photostim_onset[local_idx]) and np.isfinite(kept_photostim_duration[local_idx]):
    stim_start_abs = kept_trial_starts[local_idx] + kept_photostim_onset[local_idx]
    stim_stop_abs = stim_start_abs + kept_photostim_duration[local_idx]
    stim_on = ((abs_centers[local_idx] >= stim_start_abs) & (abs_centers[local_idx] < stim_stop_abs)).astype(np.float32)
```

iii. CONVERSION_NOTES.md: "`photostim_onset` is interpreted relative to trial start... `photostimulation_on` is 1 when the bin center falls inside the photostim interval."

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is evaluated at the same absolute bin centers as the neural data (`go_time + BIN_CENTERS_REL`), so it is inherently aligned.

ii.
```python
abs_centers = go_starts[trial_keep, None] + BIN_CENTERS_REL[None, :]
# ...
stim_on = ((abs_centers[local_idx] >= stim_start_abs) & (abs_centers[local_idx] < stim_stop_abs)).astype(np.float32)
```

iii. Same bin centers used for both neural and input data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Lick direction choice is derived from `intervals/trials/trial_instruction`, which encodes the instructed lick direction ("left" or "right"). The reference code uses `task_trial_type` ('l' or 'r'), which corresponds to `trial_instruction` in the NWB format.

ii.
```python
trial_instruction = _decode_str_array(trials["trial_instruction"]).reshape(-1)
# ...
choice_value = CHOICE_MAP[str(kept_trial_instruction[local_idx])]
# CHOICE_MAP = {"left": 0, "right": 1}
```

iii. CONVERSION_NOTES.md: "The reference code uses the paper's left/right trial label (`trial_type` in the original code path), which corresponds to `trial_instruction` in the NWB release."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The string values "left" and "right" are mapped to integers 0 and 1 respectively, as specified in the task instructions. This is broadcast to all time bins (constant per trial). Note: the reference code maps left=1, right=0, but the task instructions specify left=0, right=1, and the AI follows the instructions.

ii.
```python
CHOICE_MAP = {"left": 0, "right": 1}
choice_value = CHOICE_MAP[str(kept_trial_instruction[local_idx])]
output_trial = np.vstack([
    np.full(N_BINS, choice_value, dtype=np.int16),
    # ...
])
```

iii. CONVERSION_NOTES.md: "`choice` is encoded from `trial_instruction`: left -> 0, right -> 1."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `intervals/trials/outcome`, which contains string values "hit", "miss", or "ignore".

ii.
```python
outcomes = _decode_str_array(trials["outcome"]).reshape(-1)
# ...
outcome_value = OUTCOME_MAP[str(kept_outcomes[local_idx])]
# OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
```

iii. CONVERSION_NOTES.md: "Mapped directly from NWB `outcome`."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String outcomes are mapped to integers: ignore=0, miss=1, hit=2, matching the task instructions. This is broadcast to all time bins (constant per trial).

ii.
```python
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
outcome_value = OUTCOME_MAP[str(kept_outcomes[local_idx])]
np.full(N_BINS, outcome_value, dtype=np.int16)
```

iii. Direct mapping as specified in the instructions.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. This question refers to "Outcome" (the task does not have a "distance to reward zone" variable). Outcome is a per-trial constant broadcast to all 80 time bins, so alignment is trivial - every time bin in a trial has the same outcome value.

ii.
```python
np.full(N_BINS, outcome_value, dtype=np.int16)
```

iii. Per-trial constant, no temporal alignment needed.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from `intervals/trials/early_lick`, which contains string values "early" or "no early".

ii.
```python
early_lick = _decode_str_array(trials["early_lick"]).reshape(-1)
# ...
early_value = EARLY_LICK_MAP[str(kept_early[local_idx])]
# EARLY_LICK_MAP = {"no early": 0, "early": 1}
```

iii. CONVERSION_NOTES.md: "Mapped directly from NWB `early_lick`."

## 7-b. What processing is involved in computing `output` *Early lick*?

i. String values are mapped to integers: "no early"=0, "early"=1, matching the task instructions (no=0, yes=1). This is broadcast to all time bins (constant per trial).

ii.
```python
EARLY_LICK_MAP = {"no early": 0, "early": 1}
early_value = EARLY_LICK_MAP[str(kept_early[local_idx])]
np.full(N_BINS, early_value, dtype=np.int16)
```

iii. Direct mapping as specified in the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, using the timestamps and data columns (specifically column index 1 for y-position).

ii.
```python
tongue_group = f["acquisition/BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_timestamps = np.asarray(tongue_group["timestamps"], dtype=np.float64)
tongue_y_all = np.asarray(tongue_group["data"][:, 1], dtype=np.float32)
```

iii. CONVERSION_NOTES.md: "Side-camera tongue tracking comes from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`."

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each trial and each 50ms bin, the tongue y-position is sampled as the last camera sample within that bin (using `_bin_tongue_y`). Bins with no camera samples get a value of 0.

ii.
```python
def _bin_tongue_y(timestamps, y_values, go_time):
    bin_starts = go_time + BIN_EDGES_REL[:-1]
    bin_ends = go_time + BIN_EDGES_REL[1:]
    idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
    valid = idx >= 0
    if np.any(valid):
        valid_idx = idx[valid]
        valid[valid] &= timestamps[valid_idx] >= bin_starts[valid]
    sampled = np.zeros(N_BINS, dtype=np.float32)
    if np.any(valid):
        sampled[valid] = y_values[idx[valid]].astype(np.float32)
    return sampled
```

iii. CONVERSION_NOTES.md: "For each 50 ms neural bin, tongue y is sampled as the last camera sample within that bin."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session percentiles (40th and 60th) are computed from the entire session's tongue y-position data. Each bin's tongue y-value is then categorized: 0 if < 40th percentile, 1 if between 40th and 60th, 2 if > 60th percentile.

ii.
```python
tongue_q40, tongue_q60 = np.percentile(tongue_y_all, [40, 60])
# ...
tongue_cat = np.zeros(N_BINS, dtype=np.int16)
tongue_cat[tongue_y_trial >= tongue_q40] = 1
tongue_cat[tongue_y_trial > tongue_q60] = 2
```

iii. CONVERSION_NOTES.md: "Per-session discretization uses the full-session tongue-y distribution: < 40th percentile -> 0, 40th to 60th percentile -> 1, > 60th percentile -> 2."

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y-position is binned using the same bin edges as neural data (go_time + BIN_EDGES_REL), so it is inherently aligned. The `_bin_tongue_y` function takes the go cue time and uses the same relative bin edges.

ii.
```python
tongue_y_trial = _bin_tongue_y(tongue_timestamps, tongue_y_all, go_starts[trial_idx])
# _bin_tongue_y uses BIN_EDGES_REL, same as neural data
```

iii. Same temporal bins as neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Missing photostim values (NaN onset/duration) result in all-zero photostim signal for that trial. (2) The `_parse_optional_float_array` handles "N/A" or empty strings by returning NaN. (3) The `_resolve_sample_onsets` has fallback logic for trials where the straightforward sample onset resolution fails. (4) Sessions with zero good units are skipped. (5) All-zero neural trials are removed. (6) Tongue y bins with no camera samples default to 0.

ii.
```python
# Missing photostim:
if np.isfinite(kept_photostim_onset[local_idx]) and np.isfinite(kept_photostim_duration[local_idx]):
    # ... compute stim_on
# else stim_on stays as zeros

# N/A parsing:
def _parse_optional_float_array(dataset):
    # ...
    if value in ("N/A", "", None):
        continue
    out[i] = float(value)

# Sample onset fallback:
invalid = np.isnan(sample_onsets) | (sample_onsets < (trial_starts - 1e-9))
if np.any(invalid):
    # ... search within trial boundaries
```

iii. The AI handles edge cases defensively without crashing, defaulting to sensible values (zeros for missing data).

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is spike binning: for each good unit in each session, `_extract_unit_spikes` reads the unit's spike times, then `np.searchsorted` is called against all trial bin edges. With ~69,000 good units across 173 sessions and ~80,000+ trials, this involves many large searchsorted operations. The NWB file I/O (h5py reads) is also significant.

ii.
```python
for pos, unit_idx in enumerate(good_unit_indices):
    unit_spikes = _extract_unit_spikes(units, int(unit_idx))
    spike_bins = np.searchsorted(unit_spikes, abs_edges, side="left")
    spike_counts = np.diff(spike_bins, axis=1).astype(np.float32) / np.float32(BIN_SIZE_S)
    session_neural[:, pos, :] = spike_counts
```

iii. The per-unit loop over all good units is the inner hot loop; each iteration involves extracting spike times from HDF5 and performing searchsorted over all trial bins.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop for computing inputs/outputs (lines 274-304) could be partially vectorized. The tone_rel computation, time_from_tone, and photostimulation signals could be computed as vectorized array operations over all trials simultaneously. The `_bin_tongue_y` function is called once per trial in a loop, but could potentially be vectorized.

ii.
```python
for local_idx, trial_idx in enumerate(keep_trial_indices):
    tone_rel = kept_tone_onsets[local_idx] - go_starts[trial_idx]
    time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
    # ... photostim, tongue_y, choice, outcome, early_lick ...
```

iii. No specific justification given by the AI for this loop structure; it appears to be a straightforward implementation choice.

## 10-c. What processing does the code repeat multiple times?

i. The code reads `go_starts[trial_keep]` and `go_starts[trial_idx]` in multiple places, computing the same filtered arrays redundantly. The bin edges (`abs_edges`, `abs_centers`) are computed once as arrays but then individual rows are accessed in the per-trial loop. The `_decode_str_array` function is called separately for each string column in the trials table.

ii.
```python
# abs_edges computed for all trials at once, but then accessed per-trial:
abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
abs_centers = go_starts[trial_keep, None] + BIN_CENTERS_REL[None, :]
# Then in loop:
stim_on = ((abs_centers[local_idx] >= stim_start_abs) & ...).astype(np.float32)
```

iii. The pre-computation of `abs_edges` and `abs_centers` is actually efficient; the repeated work is minimal.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads `photostim_power` but never uses it for the binary photostimulation signal (only onset and duration are used). The `_subset_data` function creates a sample subset that is saved but may not be used by the decoder. The code also computes detailed per-session summary statistics (mean trial duration, mean tone onset, etc.) that are logged but not part of the decoder input.

ii.
```python
photostim_power = _parse_optional_float_array(trials["photostim_power"])
# photostim_power is never used after this line

# Summary statistics computed but only for logging:
"mean_trial_duration_s": float(np.mean(trial_stops[trial_keep] - trial_starts[trial_keep])),
"mean_tone_onset_rel_go_s": float(np.mean(kept_tone_onsets - go_starts[trial_keep])),
```

iii. The unused `photostim_power` read is minor overhead. The summary statistics serve documentation purposes.
