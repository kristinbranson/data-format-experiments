# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent finds every `/app/data/sub-*/*.nwb` file, sorts the paths, excludes files with no classifier-good units, and reads each selected NWB file directly with `h5py`. It loads units, trials, behavioral events, and tongue tracking session by session.

ii.
```python
def load_candidate_files() -> list[Path]:
    return sorted(DATA_DIR.glob("sub-*/*.nwb"))

with h5py.File(path, "r") as f:
    classification = decode_strings(f["units"]["classification"])
    trials = f["intervals"]["trials"]
    events = f["acquisition"]["BehavioralEvents"]
```

iii. The notes say direct HDF5 reads were chosen instead of PyNWB for speed and lower overhead. The sorted 174-file archive is treated as complete; one file is excluded because it has no classifier-good units, yielding 173 sessions.

## 1-b. How are the data split into subjects?

i. The subject ID is read from each file's NWB subject metadata, prefixed with `sub-`, and unique subjects are accumulated in first-session order. `subject_idx` maps each output session to that list.

ii.
```python
subject_id = str(f["general"]["subject"]["subject_id"][()])
if subject_id.startswith("b'"):
    subject_id = subject_id[2:-1]
subject_id = f"sub-{subject_id}"
...
if result.subject_id not in subject_to_idx:
    subject_to_idx[result.subject_id] = len(subjects)
    subjects.append(result.subject_id)
subject_idx.append(subject_to_idx[result.subject_id])
```

iii. The agent considered the NWB subject field authoritative and reports that this produces 28 subjects, matching the paper/archive.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. The session ID is derived from the filename, and output session order follows the sorted file list after removing the zero-good-unit file.

ii.
```python
session_id = path.stem.replace("_behavior+ecephys+ogen", "").replace(
    "_behavior+ecephys", ""
)
for i, path in enumerate(session_files, start=1):
    result = process_session(path, make_plot=make_plot)
    results.append(result)
```

iii. The notes state that 173 classifier-QC sessions best match the paper, better than either all 174 files or an unrelated stricter behavioral subset.

## 1-d. How are the data split into trials?

i. Trial rows come from `intervals/trials`. For each row, the agent searches behavioral events between that row's start and stop and uses the last go cue found there. One output record is made for each trial that survives filtering.

ii.
```python
for trial_idx in range(len(trial_start)):
    start = trial_start[trial_idx]
    stop = trial_stop[trial_idx]
    go_candidates = interval_values(go_times, start, stop)
    if len(go_candidates) == 0:
        raise ValueError(...)
    go_time = float(go_candidates[-1])
```

iii. The agent says matching events within trial intervals robustly handles repeated sample epochs on early-lick trials and aligns every retained trial to its go cue.

## 1-e. How are trials filtered based on quality controls?

i. A trial is kept initially only if its entire `[-2.5, +1.5]` s go-aligned window lies inside one observation interval from the first good unit. After binning, any trial whose neural tensor is entirely zero is also removed. The code does not directly filter `free_water`, and it only errors—not drops the session—if no trials remain; it does not explicitly enforce at least two retained trials.

ii.
```python
covered = np.any(
    (window_start >= session_obs_intervals[:, 0])
    & (window_end <= session_obs_intervals[:, 1])
)
if not covered:
    continue
...
nonzero_trial_mask = np.any(neural_tensor != 0, axis=(1, 2))
if n_trials_dropped_all_zero:
    neural_tensor = neural_tensor[nonzero_trial_mask]
```

iii. The notes justify full-window filtering as necessary to avoid invalid all-zero neural windows. This removed many more trials than the reference: 73,910 were retained rather than 90,860. The agent calls this lower count “by design” and says the final all-zero check resolved remaining artifacts.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times` and `spike_times_index`, restricted by `units/classification`, and is binned relative to behavioral `go_start_times`.

ii.
```python
spike_times_flat = f["units"]["spike_times"][:]
spike_times_index = f["units"]["spike_times_index"][:]
spike_times_ragged = split_ragged(spike_times_flat, spike_times_index)
good_unit_mask = classification == "good"
```

iii. The agent chose spike times because they are the raw neural representation, and chose classifier-good units to match the QC white paper and reference methods.

## 2-b. How is the `neural` data processed?

i. For each good unit, `searchsorted` counts spikes between all trial/bin edges at once. Counts are divided by 0.05 s to form firing rates in Hz, stored as `float16`. There is no smoothing, normalization, or baseline subtraction.

ii.
```python
neural_tensor = np.empty((n_trials, n_good_units, N_BINS), dtype=np.float16)
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    counts = np.diff(edge_idx, axis=1)
    neural_tensor[:, unit_idx, :] = (counts / BIN_WIDTH_S).astype(np.float16)
```

iii. The notes say firing rates, rather than counts, follow the task wording. `float16` was selected to reduce the full pickle from an otherwise much larger size; spot checks against raw counts passed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose `units.classification` equals `good` are retained. A session with no such units is excluded; `unit_quality` and individual metric thresholds are not used.

ii.
```python
classification = decode_strings(f["units"]["classification"])
good_unit_mask = classification == "good"
if int(good_unit_mask.sum()) == 0:
    return None
```

iii. The notes identify this field as the one consistent with classifier-based QC in the white paper. It yields 69,453 units in 173 sessions, close to the paper's 69,943.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The last go cue inside each trial interval is selected. Relative bin edges from -2.5 to +1.5 s are added to its absolute timestamp, and absolute spike times are counted against those edges.

ii.
```python
go_time = float(go_candidates[-1])
trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL
```

iii. The agent notes that all NWB streams share the same session clock, so adding go-relative edges is sufficient and requires no offset correction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data use 80 non-overlapping 50 ms bins across `[-2.5, +1.5)` s. Raw spikes are newly binned to that grid; there is no further rebinning.

ii.
```python
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_WIDTH_S)
```

iii. The agent explicitly deviated from the paper code's finer sliding bins because the decoder instructions require 50 ms bins and this exact window.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times`, trial start/stop bounds, and the selected go time/bin centers. The last sample-start event between trial start and go is treated as tone onset.

ii.
```python
sample_candidates = interval_values(sample_start_times, start, go_time)
sample_onset = float(sample_candidates[-1])
bin_centers_abs = go_time + BIN_CENTERS_REL
```

iii. The notes explain that early licks can replay the sample epoch, so the last sample start before go is the relevant tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Absolute bin-center times are subtracted from the selected sample onset. If no sample event exists, the code falls back to `go - 1.85 s`, although the full run reports that this fallback was never used.

ii.
```python
if len(sample_candidates) == 0:
    sample_onset = go_time - 1.85
else:
    sample_onset = float(sample_candidates[-1])
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
```

iii. The agent describes the fallback as defensive handling for missing sample events and documents that `n_missing_sample_onset_fallback = 0`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at exactly the same 80 absolute bin centers whose edges define the neural firing-rate bins.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL
```

iii. The notes report raw-to-converted checks of this time axis and state that all trials have exactly 80 go-aligned bins.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial-table `photostim_onset`, `photostim_duration`, and `photostim_power`, plus trial start and go-aligned bin centers.

ii.
```python
trial_photostim_onset = decode_strings(trials["photostim_onset"])
trial_photostim_duration = decode_strings(trials["photostim_duration"])
trial_photostim_power = decode_strings(trials["photostim_power"])
```

iii. The notes say these trial-relative fields are the authoritative stimulation epoch and that output should be a binary time series.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Optional strings are parsed as floats. If power, onset, and duration are all present, onset is converted to absolute time and each bin center is labeled 1 inside the half-open stimulation interval; otherwise all bins remain zero.

ii.
```python
if stim_power is not None and stim_onset is not None and stim_dur is not None:
    stim_start_abs = start + stim_onset
    stim_stop_abs = stim_start_abs + stim_dur
    photostim_row = ((bin_centers_abs >= stim_start_abs)
                     & (bin_centers_abs < stim_stop_abs)).astype(np.float32)
```

iii. The agent says this represents whether light is on at every requested time point and reports that spot checks against raw fields matched.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The absolute stimulation interval is tested at the same go-aligned bin centers used for the time input and corresponding neural bins.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
photostim_row = (
    (bin_centers_abs >= stim_start_abs) & (bin_centers_abs < stim_stop_abs)
).astype(np.float32)
```

iii. The agent says the shared NWB clock and common bin grid provide alignment without interpolation.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Hit/miss choices are inferred from `trial_instruction` and `outcome`. For `ignore`, the agent instead searches left/right lick event times anywhere in the full trial and uses the earliest side; if there is no lick it substitutes the instructed side.

ii.
```python
if outcome == "hit":
    return CHOICE_MAP[instruction], "instruction+outcome"
if outcome == "miss":
    opposite = "right" if instruction == "left" else "left"
    return CHOICE_MAP[opposite], "instruction+outcome"
...
return CHOICE_MAP[instruction], "ignore:instruction_fallback"
```

iii. The notes call the ignore-trial rule a “task-driven compromise” because no explicit choice exists, and acknowledge that 13,258 fallback labels inject noise. They did not create the requested `no lick` class.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded only as 0 left or 1 right and repeated across all 80 bins. The declared values are only `left` and `right`; ignored/no-response trials are forced into one of those classes.

ii.
```python
CHOICE_MAP = {"left": 0, "right": 1}
output_row[0, :] = choice_code
...
"output_values": [
    ["left", "right"],
```

iii. The agent chose a uniform `(4, 80)` output shape for convenience. It justified forced labels as enabling a two-class decoder, despite the instructions explicitly requesting `no lick`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It is read directly from the trial-table `outcome` strings.

ii.
```python
trial_outcome = decode_strings(trials["outcome"])
outcome_code = OUTCOME_MAP[trial_outcome[trial_idx]]
```

iii. The notes say the source categories exactly match the requested decoder categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. `ignore`, `miss`, and `hit` map to 0, 1, and 2, respectively, and the per-trial value is repeated over 80 bins.

ii.
```python
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
output_row[1, :] = outcome_code
```

iii. The agent says repetition gives all outputs a common time-varying array shape while preserving the per-trial label.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trial-table `early_lick` field.

ii.
```python
trial_early = decode_strings(trials["early_lick"])
early_code = EARLY_MAP[trial_early[trial_idx]]
```

iii. The notes identify this as the source's explicit early-lick label.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and `early` to 1; the value is repeated over all bins.

ii.
```python
EARLY_MAP = {"no early": 0, "early": 1}
output_row[2, :] = early_code
```

iii. The mapping is said to match the requested no/yes categories exactly.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `Camera0_side_TongueTracking` data columns x, y, and likelihood, along with that series' timestamps.

ii.
```python
tongue_data = tongue_group["data"][:].astype(np.float64)
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
xy = tongue_xyzl[:, :2]
y = tongue_xyzl[:, 1].copy()
likelihood = tongue_xyzl[:, 2]
```

iii. The notes identify this as the available tongue trace and use x/y jointly only for velocity-based outlier detection.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The agent computes 2D frame velocity, flags values above mean plus five standard deviations, replaces low-likelihood y values (`likelihood < 0.1`) with the session mean, linearly interpolates velocity/nonfinite outliers, and samples the processed trace at each bin center using the closest preceding camera frame.

ii.
```python
outlier_mask[1:] = np.isfinite(velocity) & (velocity > vel_threshold)
low_likelihood_mask = likelihood < LIKELIHOOD_THRESHOLD
processed_y[low_likelihood_mask] = session_mean_y
processed_y[interp_mask] = np.interp(...)
...
tongue_frame_idx = np.searchsorted(tongue_timestamps, bin_centers_abs, side="right") - 1
tongue_y = processed_tongue_y[tongue_frame_idx]
```

iii. The agent says mean filling and five-sigma velocity handling follow the method paper and that center/preceding-frame sampling is close to marker-alignment logic. It acknowledged mean filling causes a dominant middle class but accepted it.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The 40th and 60th percentiles are computed over every frame of the imputed/interpolated session trace. Sampled values below p40 are class 0, values from p40 through p60 are class 1, and values above p60 are class 2. No class 3 (`not visible`) is implemented.

ii.
```python
p40, p60 = np.percentile(processed_y, [40.0, 60.0])
tongue_bin = np.zeros(N_BINS, dtype=np.int16)
tongue_bin[tongue_y >= tongue_info["p40"]] = 1
tongue_bin[tongue_y > tongue_info["p60"]] = 2
```

iii. The notes say per-session percentile discretization follows the task, but deliberately treat occluded frames as the session mean. Consequently `output_values` declares only three classes and the full data are about 82.4% middle class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For every neural bin center, the code selects the most recent camera frame on the common absolute clock. Indices outside the camera range are clipped to the first/last frame. It does not average camera samples over the neural bin.

ii.
```python
tongue_frame_idx = np.searchsorted(tongue_timestamps, bin_centers_abs, side="right") - 1
tongue_frame_idx = np.clip(tongue_frame_idx, 0, len(processed_tongue_y) - 1)
tongue_y = processed_tongue_y[tongue_frame_idx]
```

iii. The agent justifies this as continuous marker alignment at bin centers and says the shared clock prevents offsets. Raw-to-converted checks verified implementation of this chosen rule.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with no good units are excluded. Trials lacking complete observation-interval coverage or having all-zero neural data are removed. Missing tone onset gets a 1.85 s pre-go fallback. Low-likelihood tongue frames are filled with session mean, velocity/nonfinite outliers are interpolated, and out-of-range camera lookup is edge-clipped. Invalid good-unit anatomy raises an error.

ii.
```python
if n_good_units == 0:
    return None
if not covered:
    continue
if len(sample_candidates) == 0:
    sample_onset = go_time - 1.85
processed_y[low_likelihood_mask] = session_mean_y
processed_y[interp_mask] = np.interp(...)
```

iii. The notes describe these as defensive rules meant to avoid fabricated zero-neural windows and follow the method paper's tongue cleanup. They report that the tone fallback was unused, while observation filtering materially reduced trial count.

## 10-a. What are the most time-consuming steps of the code?

i. The agent identifies direct NWB/HDF5 reads, per-unit spike binning, accumulation/pickling of the multi-gigabyte dataset, full decoder training, and optional plotting as the expensive work. Conversion timing was about 0.7–0.8 seconds per sampled session and the final pickle was 4.6 GB.

ii.
```python
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
...
with open(args.outpicklefile, "wb") as f:
    pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes attribute the speed to direct HDF5 access and vectorizing all trials within each unit; they projected roughly two minutes for full conversion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The outer per-trial feature-building loop and the later loop copying trial dictionaries into preallocated tensors could be vectorized or fused. The per-unit loop remains, although its trial dimension is already vectorized. Region dictionary construction and conversion from tensors to per-trial lists are also Python loops but relatively cheap.

ii.
```python
for trial_idx in range(len(trial_start)):
    ...
for keep_idx, rec in enumerate(trial_records):
    ...
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
```

iii. The notes explicitly say nested per-trial/per-neuron loops would be too slow and highlight the implemented vectorization across trials for each unit. They do not discuss vectorizing the two remaining trial loops.

## 10-c. What processing does the code repeat multiple times?

i. It reads each file twice: once in `select_files` to count good units and again in `process_session`; sample-mode timing also calls `select_files` again to estimate full runtime. Within a session it first builds per-trial dictionaries, then loops over those records to copy the same values into tensors. `go_per_trial` and `sample_on_per_trial` duplicate record fields primarily for plotting/metadata.

ii.
```python
with h5py.File(path, "r") as f:  # select_files
    classification = decode_strings(f["units"]["classification"])
...
with h5py.File(path, "r") as f:  # process_session
    classification = decode_strings(f["units"]["classification"])
...
for keep_idx, rec in enumerate(trial_records):
    input_tensor[keep_idx, 0, :] = rec["time_from_tone"]
```

iii. The agent's notes emphasize single-session bounded memory but do not acknowledge these repeated file opens and record-to-tensor copying.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It always reads left/right lick streams to manufacture ignore-trial choice labels, reads `photostim_power` only as a presence gate, computes detailed tongue velocity/outlier statistics, and creates several bookkeeping arrays/statistics not stored in the final dataset. In normal full conversion, plot payload generation is disabled, but trial dictionaries and duplicated go/sample arrays are still temporary and discarded.

ii.
```python
left_lick_times = events["left_lick_times"]["timestamps"][:]
right_lick_times = events["right_lick_times"]["timestamps"][:]
trial_photostim_power = decode_strings(trials["photostim_power"])
...
go_per_trial = np.empty(n_trials, dtype=np.float64)
sample_on_per_trial = np.empty(n_trials, dtype=np.float64)
```

iii. The notes present lick fallback, outlier processing, sanity statistics, and optional plots as validation aids. However, the ignore-choice work is unnecessary under the reference's explicit `no lick` class, and much temporary bookkeeping is not used by downstream decoding.
