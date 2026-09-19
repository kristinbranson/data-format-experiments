# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is distributed as one NWB file per session under `data/sub-<subject_id>/`. All sessions are found by globbing `sub-*/*.nwb`. Each file is opened with `h5py` (not `pynwb`) and subjects, trials, units, and behavioral events are read from the HDF5 group structure directly.

ii.
```python
nwb_paths = sorted(data_dir.glob("sub-*/*.nwb"))
...
for session_number, path in enumerate(nwb_paths, start=1):
    with h5py.File(path, "r") as f:
        trials = f["intervals/trials"]
        ...
        units = f["units"]
        ...
        be = f["acquisition/BehavioralEvents"]
```

iii. The agent initially explored using pynwb but switched to h5py for speed, stating: "I'm switching to faster HDF5-level inspection so I can find the exact inclusion rule rather than guessing which session to drop." h5py avoids pynwb overhead and UserWarnings about cached namespace versions.

## 1-b. How are the data split into subjects?

i. Each NWB file records its animal in `general/subject/subject_id`, a numeric string such as `'440956'`. That value is read for every session and used to build the subjects list and subject_idx mapping.

ii.
```python
subject = _decode_scalar(f["general"]["subject"]["subject_id"][()])
subject = str(subject)
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
subject_idx.append(subject_to_idx[subject])
```

iii. The agent used subject_id as the canonical identifier from the NWB files. Subjects are added in encounter order (order of sorted file paths) rather than sorted alphabetically.

## 1-c. How are the data split into sessions?

i. One NWB file is one session. No grouping or splitting is needed. Sessions are processed in sorted file order.

ii.
```python
nwb_paths = sorted(data_dir.glob("sub-*/*.nwb"))
...
for session_number, path in enumerate(nwb_paths, start=1):
    with h5py.File(path, "r") as f:
```

iii. The agent recognized that the dandiset stores one session per file. 174 files are found but one is dropped for having no `classification=='good'` units.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`f["intervals/trials"]`), one row per behavioural trial. The number of go cues is checked against the number of trials.

ii.
```python
trials = f["intervals/trials"]
n_trials = len(trials["id"])
...
go_starts = np.asarray(be["go_start_times"]["timestamps"], dtype=np.float64)
if len(go_starts) != n_trials:
    raise RuntimeError(f"{path.name}: expected {n_trials} go cues, found {len(go_starts)}")
```

iii. Trials are clearly defined by the trials table; the agent verified consistency with the go cue event count.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on multiple criteria:
1. `auto_water == 0` (exclude auto-water trials)
2. `free_water == 0` (exclude free-water trials)
3. `is_good_trials` mask from units (trials where good units have quality ephys)
4. `obs_intervals` mapping (match recorded trial intervals to behavioral trials)
5. All-zero neural pruning (post-hoc removal of trials where all firing rates are zero)
6. Session must have >= 2 trials remaining.

Early lick and ignore trials are kept because they are required decoder outputs.

ii.
```python
trial_keep = np.ones(n_trials, dtype=bool)
if exclude_auto_free:
    trial_keep &= auto_water == 0
    trial_keep &= free_water == 0
...
is_good_trials = np.asarray(units["is_good_trials"][good_unit_indices]).astype(bool)
...
trial_keep &= recorded_trial_mask
...
# Post-hoc all-zero removal
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
session_neural = session_neural[nonzero_trial_mask]
```

iii. The agent found that auto_water and free_water trials should be excluded because they are reward-delivery trials with no task behavior. The is_good_trials and obs_intervals filtering was added after discovering all-zero neural trials in the first conversion run. The all-zero pruning was a final sanity filter for remaining edge cases.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (sorted spike times per unit) and `units/spike_times_index` (ragged array index). Only units with `classification == 'good'` contribute.

ii.
```python
spike_times = units_group["spike_times"]
spike_index = units_group["spike_times_index"]
end = int(spike_index[unit_idx])
start = 0 if unit_idx == 0 else int(spike_index[unit_idx - 1])
return np.asarray(spike_times[start:end], dtype=np.float64)
```

iii. `spike_times` is the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms non-overlapping bins spanning [-2.5s, +1.5s] relative to the go cue (80 bins). Spike counts are computed via `np.searchsorted` on absolute bin edges, then divided by bin width to get firing rates in Hz. No smoothing, normalization, or baseline subtraction is applied.

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

iii. The agent uses the same searchsorted approach as the reference, vectorized across all trials for each unit.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept. Sessions with no good units are skipped entirely.

ii.
```python
classification = _decode_str_array(units["classification"]).reshape(-1)
...
good_unit_indices = np.flatnonzero(classification == "good")
if len(good_unit_indices) == 0:
    continue
```

iii. The agent chose `classification` over `unit_quality` after finding that classification_good had 459 units vs unit_quality_good with 1222 units in one session, recognizing that `classification` is the QC classifier described in the spike sorting QC paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are computed as absolute times by adding relative bin edges to each trial's go cue time. Since spike times and go cue times are on the same session-absolute clock, no interpolation or offset correction is needed.

ii.
```python
abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
abs_centers = go_starts[trial_keep, None] + BIN_CENTERS_REL[None, :]
```

iii. All NWB timestamps share the same global clock, so alignment is achieved by computing bin edges relative to the go cue onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins spanning [-2.5s, +1.5s] relative to go cue, giving 80 timepoints per trial. No rebinning is applied; spikes are binned directly from spike times.

ii.
```python
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_SIZE_S, dtype=np.float64)
N_BINS = len(BIN_CENTERS_REL)  # 80
```

iii. These values follow the instructions directly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` in `BehavioralEvents` (tone onset timestamps) and the go cue time. The last `sample_start` before each go cue is used as the tone onset for that trial.

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

iii. The agent recognized that early-lick trials cause sample replays, so multiple sample_start events can precede one go cue. Using `searchsorted(..., side='right') - 1` picks the last one before each go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the time from tone onset at each bin center is computed as: `BIN_CENTERS_REL - (tone_onset - go_time)`. This gives seconds since tone onset at each of the 80 bin centers.

ii.
```python
tone_rel = kept_tone_onsets[local_idx] - go_starts[trial_idx]
time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
```

iii. The tone_rel is the time of tone onset relative to go cue (negative), so subtracting it from the bin centers (also relative to go cue) gives time since tone onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin centers relative to the go cue, so they are inherently aligned. The same `BIN_CENTERS_REL` array defines both the neural bin centers and the time axis for this input.

ii.
```python
abs_centers = go_starts[trial_keep, None] + BIN_CENTERS_REL[None, :]
...
time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
```

iii. N/A

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, with `trial_starts` used to compute the absolute onset time.

ii.
```python
photostim_onset = _parse_optional_float_array(trials["photostim_onset"])
photostim_duration = _parse_optional_float_array(trials["photostim_duration"])
...
stim_start_abs = kept_trial_starts[local_idx] + kept_photostim_onset[local_idx]
stim_stop_abs = stim_start_abs + kept_photostim_duration[local_idx]
```

iii. The onset is stored as a string relative to trial start, with "N/A" for non-stimulated trials, parsed to float with NaN for N/A values.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series: 1 where the bin center falls between stim onset and offset (absolute times), 0 elsewhere. Non-stimulated trials (NaN onset/duration) remain all zeros.

ii.
```python
stim_on = np.zeros(N_BINS, dtype=np.float32)
if np.isfinite(kept_photostim_onset[local_idx]) and np.isfinite(kept_photostim_duration[local_idx]):
    stim_start_abs = kept_trial_starts[local_idx] + kept_photostim_onset[local_idx]
    stim_stop_abs = stim_start_abs + kept_photostim_duration[local_idx]
    stim_on = ((abs_centers[local_idx] >= stim_start_abs) & (abs_centers[local_idx] < stim_stop_abs)).astype(np.float32)
```

iii. The agent checks for finite values to skip non-stimulated trials.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The comparison uses `abs_centers` which are the same absolute bin centers used for neural binning, ensuring alignment.

ii.
```python
abs_centers = go_starts[trial_keep, None] + BIN_CENTERS_REL[None, :]
...
stim_on = ((abs_centers[local_idx] >= stim_start_abs) & (abs_centers[local_idx] < stim_stop_abs)).astype(np.float32)
```

iii. N/A

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trial_instruction` only (the instructed lick direction), mapped directly as left=0, right=1. It does NOT derive the animal's actual lick direction from instruction + outcome.

ii.
```python
CHOICE_MAP = {"left": 0, "right": 1}
...
choice_value = CHOICE_MAP[str(kept_trial_instruction[local_idx])]
```

iii. The agent's metadata describes this as "mapped from trial_instruction to match the paper code's left/right trial label." The agent treated `choice` as the instructed side, not the animal's actual lick choice.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The instructed direction string is mapped to an integer (left=0, right=1) and repeated across all 80 bins. There is no "no lick" category; `output_values` for choice is `["left", "right"]` (only 2 values).

ii.
```python
choice_value = CHOICE_MAP[str(kept_trial_instruction[local_idx])]
...
output_trial = np.vstack([
    np.full(N_BINS, choice_value, dtype=np.int16),
    ...
])
```

```python
"output_values": [
    ["left", "right"],
    ...
]
```

iii. The agent did not account for ignore trials (no lick) as a separate choice category.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `outcome` column of the trials table, holding strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcomes = _decode_str_array(trials["outcome"]).reshape(-1)
...
outcome_value = OUTCOME_MAP[str(kept_outcomes[local_idx])]
```

iii. The trials table stores outcome explicitly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to integers: ignore=0, miss=1, hit=2, and repeated across all 80 bins.

ii.
```python
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
...
outcome_value = OUTCOME_MAP[str(kept_outcomes[local_idx])]
np.full(N_BINS, outcome_value, dtype=np.int16)
```

iii. Straightforward mapping following the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, holding strings `'no early'` and `'early'`.

ii.
```python
early_lick = _decode_str_array(trials["early_lick"]).reshape(-1)
...
early_value = EARLY_LICK_MAP[str(kept_early[local_idx])]
```

iii. The field is available directly in the trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to integers: no early=0, early=1, repeated across 80 bins.

ii.
```python
EARLY_LICK_MAP = {"no early": 0, "early": 1}
...
early_value = EARLY_LICK_MAP[str(kept_early[local_idx])]
np.full(N_BINS, early_value, dtype=np.int16)
```

iii. Straightforward mapping.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, column 1 (tongue_y) of the 3-column data (tongue_x, tongue_y, tongue_likelihood), with timestamps.

ii.
```python
tongue_group = f["acquisition/BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_timestamps = np.asarray(tongue_group["timestamps"], dtype=np.float64)
tongue_y_all = np.asarray(tongue_group["data"][:, 1], dtype=np.float32)
```

iii. The agent confirmed the column layout from the series description attribute.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Per-session 40th and 60th percentiles are computed over ALL tongue_y values (entire session, all frames, regardless of tongue visibility/likelihood). No likelihood filtering is applied. Discretization: 0 if y < 40th pct, 1 if 40th <= y <= 60th, 2 if y > 60th. There is no "not visible" class.

ii.
```python
tongue_q40, tongue_q60 = np.percentile(tongue_y_all, [40, 60])
...
tongue_cat = np.zeros(N_BINS, dtype=np.int16)
tongue_cat[tongue_y_trial >= tongue_q40] = 1
tongue_cat[tongue_y_trial > tongue_q60] = 2
```

iii. The agent computed percentiles over all raw frames without filtering by likelihood. The agent did not implement a "not visible" class (class 3) for frames where the tongue is not detected.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories only: 0 (< 40th percentile), 1 (40th to 60th), 2 (> 60th). No fourth "not visible" category is implemented.

ii.
```python
tongue_cat = np.zeros(N_BINS, dtype=np.int16)
tongue_cat[tongue_y_trial >= tongue_q40] = 1
tongue_cat[tongue_y_trial > tongue_q60] = 2
```

```python
"output_values": [
    ...
    ["lt_40th_pct", "40th_to_60th_pct", "gt_60th_pct"],
]
```

iii. The instructions explicitly specify 4 categories including "3: not visible", but the AI only implemented 3.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The `_bin_tongue_y` function finds the last camera timestamp before each bin edge using searchsorted, checks it falls within the bin, and uses that single y value. This is a nearest-neighbor sampling approach rather than averaging all frames within each bin.

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

iii. The function uses the camera timestamps on the same global clock as spikes and go cues. It takes one frame per bin (the last one before the bin edge) rather than averaging all frames in the bin.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Session with no good units**: Skipped entirely (classification all NaN).
- **Trials without ephys data**: Excluded via is_good_trials and obs_intervals matching.
- **Auto/free water trials**: Excluded.
- **All-zero neural trials**: Removed post-hoc after binning.
- **Missing photostim values**: Parsed as NaN, treated as no stimulation.
- **Tongue bins with no frame**: Default to 0 (not handled as a special class).

ii.
```python
if len(good_unit_indices) == 0:
    continue
...
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
session_neural = session_neural[nonzero_trial_mask]
```

iii. The agent iteratively discovered edge cases (all-zero trials, is_good_trials mismatches) and added filters to handle them.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file with h5py and extracting spike times for each unit is the dominant cost. For each unit, the full spike time array slice must be read from HDF5, plus the searchsorted computation across all trials. The per-trial loop for inputs/outputs adds overhead compared to a vectorized approach.

ii.
```python
for pos, unit_idx in enumerate(good_unit_indices):
    unit_spikes = _extract_unit_spikes(units, int(unit_idx))
    spike_bins = np.searchsorted(unit_spikes, abs_edges, side="left")
```

iii. The agent chose h5py over pynwb specifically for speed.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops:
1. The per-unit loop for spike binning (reads each unit's spikes individually from HDF5 due to ragged storage).
2. The per-trial loop for computing inputs and outputs (iterates over each trial individually to compute time_from_tone, photostim, tongue_y, choice, outcome, early_lick).

The per-trial loop for inputs/outputs could be vectorized. The reference code vectorizes all of these operations.

ii.
```python
# Per-unit loop (necessary due to ragged spike times)
for pos, unit_idx in enumerate(good_unit_indices):
    unit_spikes = _extract_unit_spikes(units, int(unit_idx))
    ...

# Per-trial loop (could be vectorized)
for local_idx, trial_idx in enumerate(keep_trial_indices):
    tone_rel = kept_tone_onsets[local_idx] - go_starts[trial_idx]
    time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
    ...
```

iii. The reference solution vectorizes the input/output computation across all trials at once using broadcasting, avoiding the per-trial Python loop.

## 10-c. What processing does the code repeat multiple times?

i. The `_extract_unit_spikes` function re-reads the spike_times and spike_times_index datasets from HDF5 for each unit individually, rather than reading the full arrays once and slicing. The reference code reads the offsets and data arrays once upfront.

ii.
```python
def _extract_unit_spikes(units_group, unit_idx):
    spike_times = units_group["spike_times"]
    spike_index = units_group["spike_times_index"]
    end = int(spike_index[unit_idx])
    start = 0 if unit_idx == 0 else int(spike_index[unit_idx - 1])
    return np.asarray(spike_times[start:end], dtype=np.float64)
```

iii. Each call to `_extract_unit_spikes` accesses the HDF5 datasets, which involves repeated overhead for dataset access even if HDF5 caches data.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code reads and parses `photostim_power` (line 190) which is never used in the output. It also computes a detailed `summary` dictionary that is written to a separate JSON file but not included in the main pickle output. The `_subset_data` function creates a sample dataset that is separate from the main task requirement.

ii.
```python
photostim_power = _parse_optional_float_array(trials["photostim_power"])
...
summary = { ... }
...
sample_data = _subset_data(data, sample_sessions, args.sample_trials_per_session)
```

iii. These are minor inefficiencies; the summary and sample data were likely used for debugging during development.
