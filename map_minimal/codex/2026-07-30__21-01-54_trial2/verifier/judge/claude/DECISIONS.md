# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files under `data/sub-*/*.nwb` using `h5py.File` (not `pynwb`). Each file is opened, and trials, units, behavioral events, and tongue tracking are read from HDF5 groups directly. All 174 files are discovered via `Path.glob("sub-*/*.nwb")` and processed sequentially.

ii.
```python
nwb_paths = sorted(data_dir.glob("sub-*/*.nwb"))
...
for session_number, path in enumerate(nwb_paths, start=1):
    with h5py.File(path, "r") as f:
        trials = f["intervals/trials"]
        ...
        be = f["acquisition/BehavioralEvents"]
        go_starts = np.asarray(be["go_start_times"]["timestamps"], dtype=np.float64)
```

iii. The AI chose `h5py` over `pynwb` for direct low-level access to HDF5 groups. The trajectory shows exploration of the NWB file structure to map field names. The approach discovers all files the same way as the reference.

## 1-b. How are the data split into subjects?

i. Subject ID is read from `f["general"]["subject"]["subject_id"]` for each NWB file. Subjects are accumulated into a list as new IDs are encountered, and `subject_idx` maps each session to its subject.

ii.
```python
subject = _decode_scalar(f["general"]["subject"]["subject_id"][()])
subject = str(subject)
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
subject_idx.append(subject_to_idx[subject])
```

iii. The AI reads the same subject ID field as the reference. Subjects are in encounter order rather than sorted, but this is functionally equivalent.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session, so no further splitting is needed. The AI processes files in sorted order via `sorted(data_dir.glob(...))`.

ii.
```python
nwb_paths = sorted(data_dir.glob("sub-*/*.nwb"))
```

iii. Same as reference — one file per session, sorted glob for determinism.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials` in each NWB file. The go cue count is verified to match the trial count. Trials are indexed by `trial_starts`, `trial_stops`, etc.

ii.
```python
trials = f["intervals/trials"]
n_trials = len(trials["id"])
trial_starts = np.asarray(trials["start_time"], dtype=np.float64)
...
go_starts = np.asarray(be["go_start_times"]["timestamps"], dtype=np.float64)
if len(go_starts) != n_trials:
    raise RuntimeError(...)
```

iii. Same as reference — trials table with go-cue count verification.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three filters: (1) `auto_water == 0`, (2) `free_water == 0`, (3) trials must pass `is_good_trials` for all good units AND fall within `obs_intervals`. Additionally, after neural binning, trials with all-zero neural data are removed. A session is dropped if fewer than 2 trials survive.

ii.
```python
trial_keep = np.ones(n_trials, dtype=bool)
if exclude_auto_free:
    trial_keep &= auto_water == 0
    trial_keep &= free_water == 0
...
is_good_trials = np.asarray(units["is_good_trials"][good_unit_indices]).astype(bool)
...
recorded_trial_mask[mapped_idx[common_obs_mask]] = True
...
trial_keep &= recorded_trial_mask
...
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
session_neural = session_neural[nonzero_trial_mask]
```

iii. The AI's CONVERSION_NOTES.md states that `auto_water` and `free_water` trials are excluded, and `is_good_trials` is used for further filtering. The trajectory shows the AI discovered all-zero trials after initial conversion and added the `is_good_trials`-based filter and post-hoc zero-trial removal.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (ragged spike time array) and `units/spike_times_index` (per-unit offsets). Only units with `classification == 'good'` are used.

ii.
```python
def _extract_unit_spikes(units_group, unit_idx):
    spike_times = units_group["spike_times"]
    spike_index = units_group["spike_times_index"]
    end = int(spike_index[unit_idx])
    start = 0 if unit_idx == 0 else int(spike_index[unit_idx - 1])
    return np.asarray(spike_times[start:end], dtype=np.float64)
```

iii. Same source variables as the reference.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50 ms bins (-2.5 s to +1.5 s relative to go cue). `np.searchsorted` counts spikes in each bin, then counts are divided by the bin width (0.05 s) to get Hz. No smoothing or normalization.

ii.
```python
abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
...
for pos, unit_idx in enumerate(good_unit_indices):
    unit_spikes = _extract_unit_spikes(units, int(unit_idx))
    spike_bins = np.searchsorted(unit_spikes, abs_edges, side="left")
    spike_counts = np.diff(spike_bins, axis=1).astype(np.float32) / np.float32(BIN_SIZE_S)
    session_neural[:, pos, :] = spike_counts
```

iii. Same approach as reference: searchsorted + diff for binning, then divide by bin width for Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept. Sessions with zero good units are skipped.

ii.
```python
classification = _decode_str_array(units["classification"]).reshape(-1)
...
good_unit_indices = np.flatnonzero(classification == "good")
if len(good_unit_indices) == 0:
    continue
```

iii. Same as reference — classifier-based filtering using the `classification` field.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are computed as offsets from each trial's go cue time. All timestamps are on the same session-absolute clock, so no additional alignment is needed.

ii.
```python
abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
abs_centers = go_starts[trial_keep, None] + BIN_CENTERS_REL[None, :]
```

iii. Same as reference — go-cue-relative bin edges applied to absolute spike times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins, 80 bins total spanning -2.5 s to +1.5 s. No rebinning — spikes are binned directly from raw spike times.

ii.
```python
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_SIZE_S, dtype=np.float64)
N_BINS = len(BIN_CENTERS_REL)  # 80
```

iii. Matches instruction requirements exactly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (tone onset events) and `go_start_times` (go cue). The tone for each trial is the last `sample_start_times` entry before the go cue.

ii.
```python
sample_starts = np.asarray(be["sample_start_times"]["timestamps"], dtype=np.float64)
tone_onsets = _resolve_sample_onsets(trial_starts, go_starts, sample_starts)
```

```python
def _resolve_sample_onsets(trial_starts, go_starts, sample_starts):
    sample_idx = np.searchsorted(sample_starts, go_starts, side="right") - 1
    ...
    sample_onsets[valid] = sample_starts[sample_idx[valid]]
```

iii. Same approach as reference — last sample onset before the go cue accounts for replayed sample epochs from early licks. The AI additionally has a fallback constrained to within the trial's own time interval.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each bin center, time from tone onset = bin center time minus tone onset time (both relative to go cue).

ii.
```python
tone_rel = kept_tone_onsets[local_idx] - go_starts[trial_idx]
time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
```

iii. Same computation as reference: `CENTERS + (go - tone)` = `CENTERS - (tone - go)`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same go-cue-relative bin centers, so alignment is inherent.

ii. Both `abs_centers` for neural and `BIN_CENTERS_REL` for time_from_tone share the same grid.

iii. Same as reference.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table. Onset is interpreted as relative to trial start time.

ii.
```python
photostim_onset = _parse_optional_float_array(trials["photostim_onset"])
photostim_duration = _parse_optional_float_array(trials["photostim_duration"])
```

iii. Same source fields as reference.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is 1 if its absolute center falls between the absolute stim onset and offset, 0 otherwise. Non-stimulated trials have NaN onset/duration and remain 0.

ii.
```python
stim_start_abs = kept_trial_starts[local_idx] + kept_photostim_onset[local_idx]
stim_stop_abs = stim_start_abs + kept_photostim_duration[local_idx]
stim_on = ((abs_centers[local_idx] >= stim_start_abs) & (abs_centers[local_idx] < stim_stop_abs)).astype(np.float32)
```

iii. Same logic as reference, using absolute times rather than go-cue-relative times, which produces equivalent results.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Uses the same absolute bin centers derived from go cue times, so alignment is inherent.

ii. `abs_centers[local_idx]` is used for both neural bin edges and photostim comparison.

iii. Same as reference.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice directly from `trial_instruction` (`'left'`/`'right'`), mapping left=0 and right=1. It does NOT derive the actual lick direction from instruction x outcome.

ii.
```python
CHOICE_MAP = {"left": 0, "right": 1}
...
choice_value = CHOICE_MAP[str(kept_trial_instruction[local_idx])]
```

iii. The CONVERSION_NOTES.md says: "To preserve ignore trials and stay consistent with the reference processing, `choice` is encoded from `trial_instruction`." The AI interpreted choice as the instructed trial type rather than the animal's actual lick direction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. `trial_instruction` is mapped directly to 0 (left) or 1 (right) via a dictionary. There is no "no lick" category for ignore trials — those trials still get a left or right value based on instruction. The value is repeated across all 80 bins.

ii.
```python
choice_value = CHOICE_MAP[str(kept_trial_instruction[local_idx])]
...
output_trial = np.vstack([
    np.full(N_BINS, choice_value, dtype=np.int16),
    ...
])
```

Also `output_values` for choice: `["left", "right"]` (only 2 values, no "no lick").

iii. The AI's CONVERSION_NOTES says it uses `trial_instruction` to match the reference code's `trial_type` labeling. However, the instructions say "Lick direction choice (left = 0, right = 1, per-trial)" which implies the actual lick direction, not the instructed side.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, holding `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcomes = _decode_str_array(trials["outcome"]).reshape(-1)
...
outcome_value = OUTCOME_MAP[str(kept_outcomes[local_idx])]
```

iii. Same as reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped via dictionary: ignore=0, miss=1, hit=2. Repeated across all 80 bins.

ii.
```python
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
...
outcome_value = OUTCOME_MAP[str(kept_outcomes[local_idx])]
np.full(N_BINS, outcome_value, dtype=np.int16)
```

iii. Same as reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, holding `'no early'` and `'early'`.

ii.
```python
early_lick = _decode_str_array(trials["early_lick"]).reshape(-1)
...
early_value = EARLY_LICK_MAP[str(kept_early[local_idx])]
```

iii. Same as reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped via dictionary: `'no early'`=0, `'early'`=1. Repeated across all 80 bins.

ii.
```python
EARLY_LICK_MAP = {"no early": 0, "early": 1}
...
early_value = EARLY_LICK_MAP[str(kept_early[local_idx])]
np.full(N_BINS, early_value, dtype=np.int16)
```

iii. Same as reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, column 1 (tongue y). The AI reads the full session tongue_y array.

ii.
```python
tongue_group = f["acquisition/BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_timestamps = np.asarray(tongue_group["timestamps"], dtype=np.float64)
tongue_y_all = np.asarray(tongue_group["data"][:, 1], dtype=np.float32)
```

iii. Same source variable as reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI takes the 40th and 60th percentiles of ALL raw tongue_y values across the entire session (no likelihood filtering). It does NOT filter out frames where the tongue is not visible (low likelihood). Then for each trial bin, it samples the last camera frame within the bin (nearest-neighbor style) rather than averaging frames within the bin. The value is discretized into 3 categories based on the percentiles. There is no "not visible" category.

ii.
```python
tongue_q40, tongue_q60 = np.percentile(tongue_y_all, [40, 60])
...
def _bin_tongue_y(timestamps, y_values, go_time):
    bin_starts = go_time + BIN_EDGES_REL[:-1]
    bin_ends = go_time + BIN_EDGES_REL[1:]
    idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
    ...
    sampled[valid] = y_values[idx[valid]].astype(np.float32)
    return sampled
...
tongue_cat = np.zeros(N_BINS, dtype=np.int16)
tongue_cat[tongue_y_trial >= tongue_q40] = 1
tongue_cat[tongue_y_trial > tongue_q60] = 2
```

iii. The CONVERSION_NOTES says "Per-session discretization uses the full-session tongue-y distribution" and "tongue y is sampled as the last camera sample within that bin." The AI chose not to filter by tracking likelihood and not to handle "not visible" bins.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories: 0 (< 40th percentile), 1 (40th-60th percentile), 2 (> 60th percentile). The boundaries use `>=` for 1 and `>` for 2. There is no "not visible" (class 3) category.

ii.
```python
tongue_cat = np.zeros(N_BINS, dtype=np.int16)
tongue_cat[tongue_y_trial >= tongue_q40] = 1
tongue_cat[tongue_y_trial > tongue_q60] = 2
```

iii. The AI's `output_values` for tongue_y_position is `["lt_40th_pct", "40th_to_60th_pct", "gt_60th_pct"]` — only 3 values vs reference's 4 (which includes "not visible").

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each trial, the AI uses `searchsorted` on the tongue timestamps to find the last camera frame before each bin edge, then samples that single frame's y value. This uses the same go-cue-relative bin grid as the neural data.

ii.
```python
def _bin_tongue_y(timestamps, y_values, go_time):
    bin_starts = go_time + BIN_EDGES_REL[:-1]
    bin_ends = go_time + BIN_EDGES_REL[1:]
    idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
    ...
```

iii. The bins are aligned to the go cue just like the neural data. The sampling method (last frame) differs from the reference (mean of frames within each bin).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases:
- **Session with no good units**: Skipped (`continue` when `good_unit_indices` is empty).
- **Trials without spike data**: Filtered via `obs_intervals` + `is_good_trials` mask, plus post-hoc removal of all-zero trials.
- **Tongue tracking with no visible tongue**: Not explicitly handled — the `_bin_tongue_y` function defaults to 0.0 when no frame is found in a bin, which then gets classified as category 0 or 1 depending on the percentile values.

ii.
```python
if len(good_unit_indices) == 0:
    continue
...
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
session_neural = session_neural[nonzero_trial_mask]
...
sampled = np.zeros(N_BINS, dtype=np.float32)  # default 0.0 for missing tongue
```

iii. The AI handles missing neural data robustly but does not handle missing tongue data correctly — when the tongue is retracted (low likelihood), the y-value is still used, and when no frame exists in a bin, the default 0.0 gets discretized as a real measurement.

## 10-a. What are the most time-consuming steps of the code?

i. Reading NWB files and extracting spike times for each unit individually (one `_extract_unit_spikes` call per unit), plus the per-trial loop for computing inputs/outputs.

ii.
```python
for pos, unit_idx in enumerate(good_unit_indices):
    unit_spikes = _extract_unit_spikes(units, int(unit_idx))
    ...
for local_idx, trial_idx in enumerate(keep_trial_indices):
    ...
```

iii. The per-unit spike extraction reads the spike_times array slice for each unit individually, which involves repeated HDF5 reads rather than reading the entire buffer once.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops: (1) the per-unit spike extraction loop reads spikes one unit at a time, and (2) the per-trial loop for computing inputs, outputs, and tongue tracking. The reference code vectorizes the input/output computation across all trials.

ii.
```python
for pos, unit_idx in enumerate(good_unit_indices):
    unit_spikes = _extract_unit_spikes(units, int(unit_idx))
    ...
for local_idx, trial_idx in enumerate(keep_trial_indices):
    tone_rel = ...
    time_from_tone = ...
    stim_on = ...
    tongue_y_trial = ...
```

iii. The per-trial loop for inputs/outputs is particularly costly since each iteration constructs arrays individually rather than using vectorized operations as in the reference.

## 10-c. What processing does the code repeat multiple times?

i. The `_extract_unit_spikes` function re-reads the spike_times index for each unit call. The bin edges are computed once per session at the session level, which is efficient. However, the per-trial loop recomputes bin-related values per trial rather than vectorizing.

ii.
```python
def _extract_unit_spikes(units_group, unit_idx):
    spike_times = units_group["spike_times"]
    spike_index = units_group["spike_times_index"]
    ...
```

iii. The spike_times and spike_index datasets are accessed from the HDF5 file in every call rather than being read once.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI reads and processes `auto_water`, `photostim_power`, `trial_stops`, and `stop_time` fields that are not used in the final output. It also computes detailed per-session summary statistics that go into a JSON summary file but not into the pickle output used by the decoder.

ii.
```python
auto_water = np.asarray(trials["auto_water"], dtype=np.int64)
photostim_power = _parse_optional_float_array(trials["photostim_power"])
trial_stops = np.asarray(trials["stop_time"], dtype=np.float64)
...
summary["per_session"].append({
    "mean_trial_duration_s": float(np.mean(trial_stops[trial_keep] - trial_starts[trial_keep])),
    ...
})
```

iii. `auto_water` is used for filtering but is an extra filter not in the reference. `photostim_power` is read but never used further.
