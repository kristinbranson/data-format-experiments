# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent finds every `sub-*/*.nwb` file, sorts the paths, opens each once with `h5py`, and reads trials, behavioral events/time series, units, and subject metadata directly from the NWB HDF5 layout.

ii.
```python
nwb_paths = sorted(data_dir.glob("sub-*/*.nwb"))
for session_number, path in enumerate(nwb_paths, start=1):
    with h5py.File(path, "r") as f:
        trials = f["intervals/trials"]
        be = f["acquisition/BehavioralEvents"]
        units = f["units"]
```

iii. The trajectory says the agent established that the dataset is a collection of NWB sessions, inspected the NWB hierarchy, and chose direct `h5py` access after checking both `pynwb` and the paper repository. It reports 174 files and drops the one with no classifier-good units.

## 1-b. How are the data split into subjects?

i. A session's subject is read from `general/subject/subject_id`. Subjects are accumulated in first-seen (sorted-file) order, and each retained session receives an index into that list.

ii.
```python
subject = str(_decode_scalar(f["general"]["subject"]["subject_id"][()]))
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
subject_idx.append(subject_to_idx[subject])
```

iii. The agent treated the NWB subject field as the canonical identifier and verified that the release contains 28 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session; output session order is sorted path order. Sessions with no good units or fewer than two surviving trials are omitted.

ii.
```python
nwb_paths = sorted(data_dir.glob("sub-*/*.nwb"))
for session_number, path in enumerate(nwb_paths, start=1):
    with h5py.File(path, "r") as f:
        ...
    if len(good_unit_indices) == 0:
        continue
```

iii. The trajectory explicitly identifies the files as NWB sessions and documents that one of 174 files has no `classification == "good"` units, yielding 173 output sessions.

## 1-d. How are the data split into trials?

i. Rows of `intervals/trials` define trials. The agent checks that `go_start_times` has exactly one timestamp per row, constructs one aligned neural/input/output item per retained row, and preserves trial order.

ii.
```python
n_trials = len(trials["id"])
go_starts = np.asarray(be["go_start_times"]["timestamps"], dtype=np.float64)
if len(go_starts) != n_trials:
    raise RuntimeError(...)
keep_trial_indices = np.flatnonzero(trial_keep)
```

iii. The agent inspected event counts and recognized that sample events can repeat after early licks, whereas go cues map one-to-one to trial-table rows.

## 1-e. How are trials filtered based on quality controls?

i. By default the agent excludes both `auto_water` and `free_water`. It then requires every retained good unit's `is_good_trials` flag to be true over the mapped observation intervals, removes any residual trial whose entire neural tensor is zero, and drops sessions with fewer than two trials. Early-lick, ignore, and photostimulation trials are retained.

ii.
```python
trial_keep &= auto_water == 0
trial_keep &= free_water == 0
common_obs_mask = np.all(is_good_trials[:, :n_recorded_trials], axis=0)
recorded_trial_mask[mapped_idx[common_obs_mask]] = True
trial_keep &= recorded_trial_mask
nonzero_trial_mask = np.any(session_neural != 0, axis=(1, 2))
```

iii. The trajectory says this was intended to follow the paper's “regular trial” and common-good-ephys rules while retaining requested decoder labels. Its notes explicitly justify excluding auto/free-water trials and using `obs_intervals` plus `is_good_trials`; this produced 89,068 trials, versus 90,860 in the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times`, indexed by `spike_times_index`, for units whose `classification` is `good`; go-cue timestamps define trial-relative bin edges.

ii.
```python
classification = _decode_str_array(units["classification"]).reshape(-1)
good_unit_indices = np.flatnonzero(classification == "good")
unit_spikes = _extract_unit_spikes(units, int(unit_idx))
abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
```

iii. The agent confirmed the NWB files contain the post-QC labels described by the paper and chose the spike-time representation needed to compute requested rates.

## 2-b. How is the `neural` data processed?

i. For each good unit, `searchsorted` counts spikes between all adjacent absolute edges. Counts are divided by 0.05 seconds to yield unsmoothed `float32` firing rates in Hz.

ii.
```python
spike_bins = np.searchsorted(unit_spikes, abs_edges, side="left")
spike_counts = np.diff(spike_bins, axis=1).astype(np.float32) / np.float32(BIN_SIZE_S)
session_neural[:, pos, :] = spike_counts
```

iii. The trajectory and metadata state that non-overlapping 50-ms spike counts are converted to Hz, matching the paper helper's rate behavior.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units labeled `classification == "good"` are included; no individual metric thresholds or `unit_quality` filter are added. A session with no such units is dropped. Trial-level `is_good_trials` is additionally intersected across all included units.

ii.
```python
good_unit_indices = np.flatnonzero(classification == "good")
if len(good_unit_indices) == 0:
    continue
is_good_trials = np.asarray(units["is_good_trials"][good_unit_indices]).astype(bool)
common_obs_mask = np.all(is_good_trials[:, :n_recorded_trials], axis=0)
```

iii. The agent investigated `classification`, `unit_quality`, and paper QC, concluding that `classification` is the post-QC classifier verdict. It added common trial validity because it interpreted paper preprocessing as requiring common ephys support.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Relative edges from -2.5 to +1.5 seconds are added to each trial's absolute go-cue timestamp, and spikes are counted within those absolute intervals.

ii.
```python
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_SIZE_S)
abs_edges = go_starts[trial_keep, None] + BIN_EDGES_REL[None, :]
```

iii. The agent notes that NWB spike and event timestamps share a session clock, so no interpolation or extra offset is required.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 50 ms: 80 non-overlapping bins cover the four-second window. Raw spike times are binned once; no smoothing or later temporal rebinning is performed.

ii.
```python
BIN_SIZE_S = 0.05
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + (BIN_SIZE_S / 2.0)
N_BINS = len(BIN_CENTERS_REL)
```

iii. This directly follows the decoder instruction, and the agent verified every output trial has 80 bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times`, `go_start_times`, trial start times, and the common go-relative bin centers. The final sample onset before each go cue is selected.

ii.
```python
sample_starts = np.asarray(be["sample_start_times"]["timestamps"], dtype=np.float64)
tone_onsets = _resolve_sample_onsets(trial_starts, go_starts, sample_starts)
```

iii. The agent discovered that early licks can replay sample epochs, so a naive one-event-per-trial mapping is invalid; its notes say the last sample before go is the operative tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. `searchsorted(..., side="right") - 1` finds the last sample event at or before go, with a trial-bounded fallback. For each bin, time since tone is bin-center time minus the tone's go-relative offset.

ii.
```python
sample_idx = np.searchsorted(sample_starts, go_starts, side="right") - 1
tone_rel = kept_tone_onsets[local_idx] - go_starts[trial_idx]
time_from_tone = (BIN_CENTERS_REL - tone_rel).astype(np.float32)
```

iii. The fallback was included to prevent assigning a sample event from another trial, while otherwise implementing the final-pre-go decision.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the identical go-cue-relative 50-ms bin centers; each time value denotes elapsed seconds at the center of the corresponding neural bin.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + (BIN_SIZE_S / 2.0)
time_from_tone = BIN_CENTERS_REL - tone_rel
```

iii. The agent describes all streams as placed on the same go-aligned grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial `start_time`, `photostim_onset`, and `photostim_duration`, plus absolute centers derived from each go cue. `photostim_power` is read but not used.

ii.
```python
photostim_onset = _parse_optional_float_array(trials["photostim_onset"])
photostim_duration = _parse_optional_float_array(trials["photostim_duration"])
stim_start_abs = kept_trial_starts[local_idx] + kept_photostim_onset[local_idx]
```

iii. The agent interpreted onset as relative to trial start and used duration to construct the active interval, consistent with the NWB timing inspection.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. String `N/A` values become NaN. For valid trials, a bin is 1 if its center is at or after stimulation onset and before stimulation offset; otherwise it is 0.

ii.
```python
stim_on = np.zeros(N_BINS, dtype=np.float32)
if np.isfinite(...) and np.isfinite(...):
    stim_stop_abs = stim_start_abs + kept_photostim_duration[local_idx]
    stim_on = ((abs_centers[local_idx] >= stim_start_abs) &
               (abs_centers[local_idx] < stim_stop_abs)).astype(np.float32)
```

iii. The trajectory says a time-varying binary series was needed rather than a per-trial stimulation flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation bounds are compared with the absolute centers of the same go-aligned bins used for neural data.

ii.
```python
abs_centers = go_starts[trial_keep, None] + BIN_CENTERS_REL[None, :]
stim_on = ((abs_centers[local_idx] >= stim_start_abs) &
           (abs_centers[local_idx] < stim_stop_abs)).astype(np.float32)
```

iii. This places the stimulation state and firing rate on matching time indices without interpolation.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The agent derives “choice” solely from `trial_instruction`, using the instructed left/right side; it does not use outcome and therefore has no no-lick class.

ii.
```python
CHOICE_MAP = {"left": 0, "right": 1}
choice_value = CHOICE_MAP[str(kept_trial_instruction[local_idx])]
```

iii. The agent explicitly reasoned that paper code's `trial_type` corresponds to NWB `trial_instruction` and chose consistency with that label. Its notes acknowledge the requested term is lick direction but do not derive actual behavior.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Left is mapped to 0 and right to 1, then the value is repeated across all 80 bins. `output_values` contains only `left` and `right`.

ii.
```python
np.full(N_BINS, choice_value, dtype=np.int16)
...
["left", "right"],
```

iii. The agent used a time-constant output array to share the common output shape; it justified the two categories by treating choice as the instructed trial label.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials table's `outcome` strings.

ii.
```python
outcomes = _decode_str_array(trials["outcome"]).reshape(-1)
outcome_value = OUTCOME_MAP[str(kept_outcomes[local_idx])]
```

iii. The field already contains exactly the requested three categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. `ignore`, `miss`, and `hit` map to 0, 1, and 2, respectively, and the per-trial value is repeated over 80 bins.

ii.
```python
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
np.full(N_BINS, outcome_value, dtype=np.int16)
```

iii. The agent selected the order specified by the requested output categories and used repetition for a uniform time-shaped output.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It is read directly from the trials table's `early_lick` field.

ii.
```python
early_lick = _decode_str_array(trials["early_lick"]).reshape(-1)
```

iii. The agent found explicit `no early`/`early` values, so no inference from lick timestamps was needed.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and `early` to 1; the result is repeated across all bins.

ii.
```python
EARLY_LICK_MAP = {"no early": 0, "early": 1}
np.full(N_BINS, early_value, dtype=np.int16)
```

iii. This implements the requested no/yes categories while maintaining the shared output matrix shape.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses column 1 of `Camera0_side_TongueTracking/data` and that series' timestamps. It does not read or use column 2, the tongue-likelihood/visibility signal.

ii.
```python
tongue_group = f["acquisition/BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_timestamps = np.asarray(tongue_group["timestamps"], dtype=np.float64)
tongue_y_all = np.asarray(tongue_group["data"][:, 1], dtype=np.float32)
```

iii. The agent identified the side-camera tongue series as the requested measurement, but its notes describe only y and omit the visibility channel.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The agent calculates 40th/60th percentiles over all raw session y frames. For each neural bin it takes the last camera sample inside that bin; bins without a sample are filled with zero. It performs no likelihood filtering and no within-bin averaging.

ii.
```python
tongue_q40, tongue_q60 = np.percentile(tongue_y_all, [40, 60])
idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
sampled = np.zeros(N_BINS, dtype=np.float32)
sampled[valid] = y_values[idx[valid]].astype(np.float32)
```

iii. The agent claimed reference alignment uses the most recent marker sample rather than interpolation and chose full-session raw-frame percentiles. It did not discuss the tracker likelihood or required not-visible category.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Values below the 40th percentile remain 0; values at least the 40th percentile become 1; values strictly above the 60th percentile become 2. No category 3 is produced, and `output_values` lists only three classes.

ii.
```python
tongue_cat = np.zeros(N_BINS, dtype=np.int16)
tongue_cat[tongue_y_trial >= tongue_q40] = 1
tongue_cat[tongue_y_trial > tongue_q60] = 2
...
["lt_40th_pct", "40th_to_60th_pct", "gt_60th_pct"],
```

iii. The agent followed the percentile boundaries for visible coordinates but omitted the explicitly requested `3: not visible` decision.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each go-aligned 50-ms interval, `_bin_tongue_y` selects the last timestamped y sample that falls inside that interval, yielding 80 corresponding values.

ii.
```python
bin_starts = go_time + BIN_EDGES_REL[:-1]
bin_ends = go_time + BIN_EDGES_REL[1:]
idx = np.searchsorted(timestamps, bin_ends, side="left") - 1
valid[valid] &= timestamps[valid_idx] >= bin_starts[valid]
```

iii. The agent reasoned that camera and neural timestamps share an absolute clock and that last-sample assignment matched a preprocessing helper.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. `N/A` optional floats become NaN; missing tone matches trigger a bounded fallback or error; sessions with no good units and sessions with fewer than two trials are skipped; unrecorded/common-bad and all-zero neural trials are removed. However, absent tongue samples are silently represented as y=0 and then a visible low category, and unlabelled unit text is not specially normalized beyond decoding.

ii.
```python
if value in ("N/A", "", None):
    continue
...
if len(candidates) == 0:
    raise RuntimeError(...)
sampled = np.zeros(N_BINS, dtype=np.float32)
```

iii. The trajectory emphasizes checks, summary counts, and dropping fabricated all-zero neural data. It does not recognize low-confidence or absent tongue tracking as missing data, despite the requested explicit category.

## 10-a. What are the most time-consuming steps of the code?

i. The likely dominant conversion work is reading 174 large NWB files, extracting ragged spikes, performing `searchsorted` for every good unit, accumulating a very large in-memory result, and pickling it. The agent also voluntarily ran full conversion twice and trained the decoder for 200 epochs, which dominated its overall trajectory.

ii.
```python
for session_number, path in enumerate(nwb_paths, start=1):
    with h5py.File(path, "r") as f:
        ...
        for pos, unit_idx in enumerate(good_unit_indices):
            unit_spikes = _extract_unit_spikes(units, int(unit_idx))
            spike_bins = np.searchsorted(unit_spikes, abs_edges, side="left")
with open(args.full_out, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The agent monitored conversion progress and spent a large part of the trajectory waiting for full decoder training. It did not provide a measured conversion-stage profiling breakdown.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Neural work is vectorized over trials but loops over ragged units. Input/output/tongue construction loops over every retained trial; brain-region mapping and sample-subset region remapping also loop item-by-item. Trial construction could be vectorized substantially; the per-unit loop is harder to remove because spike arrays are ragged.

ii.
```python
for pos, unit_idx in enumerate(good_unit_indices):
    ...
for local_idx, trial_idx in enumerate(keep_trial_indices):
    ...
for i, region_idx in enumerate(data["brain_region_idx"][sess_idx]):
    ...
```

iii. The trajectory focused on correctness and completing the full dataset rather than profiling or refactoring these loops. Its vectorized `searchsorted` across all trials is an intentional efficiency choice.

## 10-c. What processing does the code repeat multiple times?

i. Inside conversion, each trial repeatedly builds small constant arrays with `np.full`/`vstack` and calls the tongue bin sampler; name-to-index dictionaries are checked repeatedly. Outside the core pass, `main` also traverses/copies selected full-data objects to make a sample and serializes both full and sample datasets. The trajectory ran the complete conversion more than once during validation.

ii.
```python
for local_idx, trial_idx in enumerate(keep_trial_indices):
    tongue_y_trial = _bin_tongue_y(...)
    output_trial = np.vstack([np.full(N_BINS, choice_value, ...), ...])
...
sample_data = _subset_data(data, sample_sessions, args.sample_trials_per_session)
```

iii. The agent generated the sample and summaries to support verifier/decoder sanity checks, and repeated full conversion after changes to ensure final artifacts reflected the latest code.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It parses `photostim_power` but never uses it; computes several summary-only quantities (including mean trial duration, mean tone offset, per-session quantiles, and counts); creates and writes a sample dataset and conversion summary not required by the target output; and stores verbose bin-edge/center metadata not used by the decoder.

ii.
```python
photostim_power = _parse_optional_float_array(trials["photostim_power"])
...
"mean_trial_duration_s": float(np.mean(...)),
"mean_tone_onset_rel_go_s": float(np.mean(...)),
...
with open(args.sample_out, "wb") as f:
    pickle.dump(sample_data, f, ...)
```

iii. The trajectory says these extra artifacts and statistics were created for reproducibility, sanity checks, and sample/full decoder validation, rather than because downstream analysis required them.
