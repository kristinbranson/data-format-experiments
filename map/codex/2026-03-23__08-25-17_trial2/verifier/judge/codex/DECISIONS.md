# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all candidate sessions by globbing `sub-*/*.nwb` under `/app/data`, keeps only files with at least one `units.classification == "good"` unit, and then opens each NWB file with `h5py`. Within each file it reads subject metadata, unit metadata and spike times, the trial table, behavioral event timestamps, and continuous tongue-tracking data.

ii. ```python
def load_candidate_files() -> list[Path]:
    return sorted(DATA_DIR.glob("sub-*/*.nwb"))

def select_files(all_files: list[Path], sample_mode: bool) -> tuple[list[Path], list[str]]:
    selected: list[Path] = []
    excluded: list[str] = []
    for path in all_files:
        with h5py.File(path, "r") as f:
            classification = decode_strings(f["units"]["classification"])
            if np.sum(classification == "good") == 0:
                excluded.append(path.name)
                continue
        selected.append(path)
```

```python
with h5py.File(path, "r") as f:
    classification = decode_strings(f["units"]["classification"])
    anno_name = decode_strings(f["units"]["anno_name"])
    subject_id = str(f["general"]["subject"]["subject_id"][()])
    spike_times_flat = f["units"]["spike_times"][:]
    trials = f["intervals"]["trials"]
    events = f["acquisition"]["BehavioralEvents"]
    tongue_group = f["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
```

iii. In `CONVERSION_NOTES.md`, the agent says it used direct NWB reads with `h5py` for speed and treated the NWB archive as the local equivalent of the paper’s processed session files, excluding the single session with zero classifier-good units.

## 1-b. How are the data split into subjects?

i. Subjects are split by `general/subject/subject_id` from each NWB file. The agent normalizes this to strings like `sub-440956`, builds a unique `subjects` list in first-seen order, and stores one `subject_idx` entry per session.

ii. ```python
subject_id = str(f["general"]["subject"]["subject_id"][()])
if subject_id.startswith("b'"):
    subject_id = subject_id[2:-1]
subject_id = f"sub-{subject_id}"
```

```python
if result.subject_id not in subject_to_idx:
    subject_to_idx[result.subject_id] = len(subjects)
    subjects.append(result.subject_id)
subject_idx.append(subject_to_idx[result.subject_id])
```

iii. The agent’s notes say subject IDs should come directly from NWB subject metadata and that session order should follow the sorted NWB file list.

## 1-c. How are the data split into sessions?

i. The agent treats each NWB file as one session. After excluding files with zero classifier-good units, it processes the remaining files one by one and appends one session entry each to `neural`, `input`, `output`, and `brain_region_idx`.

ii. ```python
all_files = load_candidate_files()
session_files, excluded_sessions = select_files(all_files, sample_mode=sample_mode)

for i, path in enumerate(session_files, start=1):
    result = process_session(path, make_plot=make_plot)
    if result is None:
        print("  skipped (no good units)")
        continue
    results.append(result)
```

```python
return {
    "neural": neural,
    "input": decoder_input,
    "output": output,
    "subjects": subjects,
    "subject_idx": np.asarray(subject_idx, dtype=np.int32),
    "brain_region_idx": brain_region_idx,
}
```

iii. In the notes, the agent explicitly chose a “173-session QC-filtered dataset,” interpreting one NWB file as one behavioral session and excluding the one archive file with no good units.

## 1-d. How are the data split into trials?

i. Trials are split using rows of `intervals/trials`. For each row, the agent finds the go cue inside that trial’s `[start_time, stop_time]`, builds a per-trial go-aligned window, and stores one `(neurons, time)` matrix plus matching input/output arrays for that trial.

ii. ```python
trials = f["intervals"]["trials"]
trial_start = trials["start_time"][:].astype(np.float64)
trial_stop = trials["stop_time"][:].astype(np.float64)
```

```python
for trial_idx in range(len(trial_start)):
    start = trial_start[trial_idx]
    stop = trial_stop[trial_idx]
    go_candidates = interval_values(go_times, start, stop)
    if len(go_candidates) == 0:
        raise ValueError(f"{path.name}: no go cue found for trial {trial_idx}")
    go_time = float(go_candidates[-1])
```

iii. The notes describe trial handling as “event-in-trial matching rather than row order assumptions,” with one converted trial per kept row of the NWB trial table.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not apply the paper’s stricter analysis-only “regular trial” mask. Instead, it keeps early-lick, ignore, and photostimulation trials because they are decoder targets/inputs, but drops trials whose full `[-2.5, 1.5] s` go-aligned window is not fully covered by the unit observation intervals and then drops any remaining all-zero neural trials.

ii. ```python
window_start = go_time + WINDOW_START_S
window_end = go_time + WINDOW_END_S
covered = np.any(
    (window_start >= session_obs_intervals[:, 0])
    & (window_end <= session_obs_intervals[:, 1])
)
if not covered:
    n_trials_dropped_outside_obs += 1
    continue
```

```python
nonzero_trial_mask = np.any(neural_tensor != 0, axis=(1, 2))
if n_trials_dropped_all_zero:
    neural_tensor = neural_tensor[nonzero_trial_mask]
    input_tensor = input_tensor[nonzero_trial_mask]
    output_tensor = output_tensor[nonzero_trial_mask]
```

iii. The agent’s notes justify this as a task-specific compromise: reference analyses often exclude early/ignore/stim trials, but the decoder task explicitly asks to preserve those variables, so it only removes trials with invalid neural coverage.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `units/spike_times` and `units/spike_times_index` for units whose `units/classification` is `"good"`. The agent also reads `units/obs_intervals` and `units/obs_intervals_index` to decide which trials have valid neural coverage.

ii. ```python
classification = decode_strings(f["units"]["classification"])
good_unit_mask = classification == "good"
spike_times_flat = f["units"]["spike_times"][:]
spike_times_index = f["units"]["spike_times_index"][:]
spike_times_ragged = split_ragged(spike_times_flat, spike_times_index)
good_unit_indices = np.flatnonzero(good_unit_mask)
good_spike_times = [np.asarray(spike_times_ragged[i], dtype=np.float64) for i in good_unit_indices]
obs_intervals = f["units"]["obs_intervals"][:].astype(np.float64)
obs_intervals_index = f["units"]["obs_intervals_index"][:]
```

iii. The agent’s Step 5 notes map `units.spike_times` for classifier-good units directly to `neural` and describe `units.classification == "good"` as the local equivalent of the paper’s external QC good-unit lists.

## 2-b. How is the `neural` data processed?

i. The agent converts absolute spike times into per-trial go-aligned firing rates. It builds absolute bin edges for each kept trial, counts spikes per 50 ms bin via `np.searchsorted`, divides by 0.05 s to get Hz, stores the session tensor as `float16`, and then converts it to a list of trial matrices.

ii. ```python
trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL
```

```python
neural_tensor = np.empty((n_trials, n_good_units, N_BINS), dtype=np.float16)
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    counts = np.diff(edge_idx, axis=1)
    neural_tensor[:, unit_idx, :] = (counts / BIN_WIDTH_S).astype(np.float16)
```

iii. In the notes, the agent says the neural representation should be firing rates rather than counts because the instructions explicitly ask for 50 ms bins “for computing firing rates.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neural data are filtered by keeping only units with `classification == "good"`, excluding sessions with zero such units, and keeping only trials with full observation-interval coverage plus nonzero neural activity. It also requires nonempty anatomical labels for the kept units.

ii. ```python
good_unit_mask = classification == "good"
n_good_units = int(good_unit_mask.sum())
if n_good_units == 0:
    return None
```

```python
def build_region_labels(anno_name: np.ndarray, good_unit_mask: np.ndarray) -> list[str]:
    labels = [str(x) for x in anno_name[good_unit_mask].tolist()]
    if any((label == "" or label.lower() == "nan") for label in labels):
        raise ValueError("Good units unexpectedly contain empty anatomical annotations.")
    return labels
```

iii. The notes argue that `units.classification == "good"` matches the paper’s classifier-based QC better than `unit_quality == "good"`, and that the one session with zero good units should be dropped to recover the paper’s 173 sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. Each trial is aligned to go cue onset. The agent finds the go event inside the trial interval, takes that time as zero, and bins spikes from 2.5 s before to 1.5 s after go cue.

ii. ```python
go_candidates = interval_values(go_times, start, stop)
go_time = float(go_candidates[-1])
window_start = go_time + WINDOW_START_S
window_end = go_time + WINDOW_END_S
bin_centers_abs = go_time + BIN_CENTERS_REL
```

iii. The notes repeatedly say the reference code uses a “time to go” convention and that go-cue alignment should be preserved while only the final bin width/window are changed for the decoder task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use non-overlapping 50 ms bins over `[-2.5, 1.5)`, yielding 80 bins. The agent does not first compute the paper’s 40 ms / 3.4 ms sliding-rate representation and then rebin; it bins the raw spikes directly into the decoder bins.

ii. ```python
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_WIDTH_S, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_WIDTH_S / 2.0
N_BINS = len(BIN_CENTERS_REL)
```

iii. The notes call this an explicit task-driven deviation: keep the reference alignment and QC logic, but intentionally change the final representation to 50 ms bins because the instructions require it.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The time-from-tone input is derived from `BehavioralEvents/sample_start_times/timestamps`, `BehavioralEvents/go_start_times/timestamps`, and the trial start/stop times in `intervals/trials`.

ii. ```python
go_times = events["go_start_times"]["timestamps"][:].astype(np.float64)
sample_start_times = events["sample_start_times"]["timestamps"][:].astype(np.float64)
```

```python
sample_candidates = interval_values(sample_start_times, start, go_time)
if len(sample_candidates) == 0:
    sample_onset = go_time - 1.85
else:
    sample_onset = float(sample_candidates[-1])
```

iii. The agent’s notes justify using the last sample-start before go cue because early licks can replay the sample/delay epoch, so the last sample onset is the one that actually leads into the observed go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each kept trial, the agent finds the last sample onset before go cue, computes the absolute neural bin centers for that trial, and subtracts the sample onset from each bin center. If no sample event is found, it falls back to `go_time - 1.85`.

ii. ```python
bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
```

iii. The notes describe the fallback as a minor-mistake handler and report that no trial ultimately needed it (`n_missing_sample_onset_fallback = 0`).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at exactly the same 80 go-aligned bin centers used for neural firing rates, so each trial’s tone-time input is already on the neural time base.

ii. ```python
bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
input_tensor[keep_idx, 0, :] = rec["time_from_tone"]
```

iii. The notes explicitly say this input should be stored as a time-varying signal on the decoder time axis rather than as a single event time.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the per-trial NWB trial-table fields `photostim_onset`, `photostim_duration`, `photostim_power`, plus `start_time` so the trial-relative stimulation fields can be converted to absolute times.

ii. ```python
trial_photostim_onset = decode_strings(trials["photostim_onset"])
trial_photostim_duration = decode_strings(trials["photostim_duration"])
trial_photostim_power = decode_strings(trials["photostim_power"])
```

iii. The notes map these fields to the reference code’s `task_stimulation` representation and choose the trial table rather than separate event streams as the primary source.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The agent parses the trial-table strings, treats `N/A` as missing, converts trial-relative onset and duration to absolute time using `start_time`, and marks each neural bin center as 1 if it falls within the stimulation interval and 0 otherwise.

ii. ```python
stim_onset = parse_optional_float(trial_photostim_onset[trial_idx])
stim_dur = parse_optional_float(trial_photostim_duration[trial_idx])
stim_power = parse_optional_float(trial_photostim_power[trial_idx])
if stim_power is not None and stim_onset is not None and stim_dur is not None:
    stim_start_abs = start + stim_onset
    stim_stop_abs = stim_start_abs + stim_dur
    photostim_row = (
        (bin_centers_abs >= stim_start_abs) & (bin_centers_abs < stim_stop_abs)
    ).astype(np.float32)
```

iii. The agent’s notes say trials with `photostim_power == N/A` become all-zero control trials and that the binary series is evaluated at the 50 ms decoder bins.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is aligned on the same go-cue-centered bin centers as the neural data. The agent computes absolute bin-center times from each trial’s go cue and checks whether each center falls in the stimulation interval.

ii. ```python
bin_centers_abs = go_time + BIN_CENTERS_REL
photostim_row = (
    (bin_centers_abs >= stim_start_abs) & (bin_centers_abs < stim_stop_abs)
).astype(np.float32)
input_tensor[keep_idx, 1, :] = rec["photostim_on"]
```

iii. The notes describe this as preserving the reference alignment logic while changing only the final output bin width.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trial_instruction`, `outcome`, and, for ignore trials, from the behavioral event streams `left_lick_times` and `right_lick_times` within the trial interval.

ii. ```python
trial_instruction = decode_strings(trials["trial_instruction"])
trial_outcome = decode_strings(trials["outcome"])
left_lick_times = events["left_lick_times"]["timestamps"][:].astype(np.float64)
right_lick_times = events["right_lick_times"]["timestamps"][:].astype(np.float64)
```

iii. The notes say there is no explicit per-trial choice field in the raw data, so hit/miss choice must be inferred and ignore trials require a fallback.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For `hit` trials, choice is the instructed side. For `miss` trials, choice is the opposite side. For `ignore` trials, the agent uses the earliest lick side found inside the trial, and if there is no lick it falls back to the instructed side. The resulting scalar label is repeated across all 80 bins.

ii. ```python
def infer_choice(...):
    if outcome == "hit":
        return CHOICE_MAP[instruction], "instruction+outcome"
    if outcome == "miss":
        opposite = "right" if instruction == "left" else "left"
        return CHOICE_MAP[opposite], "instruction+outcome"
    ...
    return CHOICE_MAP[instruction], "ignore:instruction_fallback"
```

```python
choice_code, choice_source = infer_choice(...)
output_row[0, :] = choice_code
```

iii. The agent’s notes call ignore-trial choice assignment “the main task-specific edge case” and document it as an unavoidable placeholder because the source data do not contain a native choice label on no-response trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived directly from the NWB trial-table field `intervals/trials/outcome`.

ii. ```python
trial_outcome = decode_strings(trials["outcome"])
outcome_code = OUTCOME_MAP[trial_outcome[trial_idx]]
```

iii. The notes say this variable is a direct mapping from the raw trial table to the requested categorical decoder output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The agent maps outcome strings to integers (`ignore=0`, `miss=1`, `hit=2`) and broadcasts that per-trial value across all 80 time bins.

ii. ```python
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
...
output_row[1, :] = outcome_code
```

iii. The notes state that this matches the user specification exactly and uses a time-varying array only because the decoder format expects a common per-bin shape.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. The agent does not compute any `Distance to reward zone` output, because that variable is not part of the instructions or its output schema. Instead, it aligns the implemented `outcome` output by repeating the trial label across the same 80 go-aligned neural bins.

ii. ```python
output_row = np.empty((4, N_BINS), dtype=np.int16)
output_row[1, :] = outcome_code
```

iii. The mismatch comes from the question text, not the code: the implemented outputs are `choice`, `outcome`, `early_lick`, and `tongue_y_bin`, exactly as listed in the agent’s metadata.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from `intervals/trials/early_lick`.

ii. ```python
trial_early = decode_strings(trials["early_lick"])
early_code = EARLY_MAP[trial_early[trial_idx]]
```

iii. The notes treat this as another direct raw-trial-table mapping required by the decoder task.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The agent maps `no early -> 0` and `early -> 1`, then repeats that label across all 80 bins for the trial.

ii. ```python
EARLY_MAP = {"no early": 0, "early": 1}
...
output_row[2, :] = early_code
```

iii. The notes say this matches the user specification exactly and keeps the common time-varying output shape.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking/data` and its `timestamps`. The agent uses the second data column as y-position and the third column as DeepLabCut likelihood.

ii. ```python
tongue_group = f["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_data = tongue_group["data"][:].astype(np.float64)
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
```

```python
xy = tongue_xyzl[:, :2].astype(np.float64, copy=False)
y = tongue_xyzl[:, 1].astype(np.float64, copy=True)
likelihood = tongue_xyzl[:, 2].astype(np.float64, copy=False)
```

iii. The notes say the local NWB side-view tongue trajectory is the directly relevant marker stream for this task.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The agent preprocesses the session-wide tongue trace before trialization: it flags low-likelihood frames (`< 0.1`), computes frame-to-frame tongue speed from x/y coordinates, marks velocity outliers above `mean + 5*std`, fills low-likelihood frames with the session mean y, linearly interpolates outliers and NaNs, and falls back to a constant session mean if nothing is valid.

ii. ```python
dt = np.diff(tongue_timestamps)
velocity = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
vel_threshold = vel_mean + 5.0 * vel_std
outlier_mask[1:] = np.isfinite(velocity) & (velocity > vel_threshold)
low_likelihood_mask = likelihood < LIKELIHOOD_THRESHOLD
```

```python
processed_y[low_likelihood_mask] = session_mean_y
interp_mask = outlier_mask | ~np.isfinite(processed_y)
if np.any(interp_mask):
    processed_y[interp_mask] = np.interp(
        tongue_timestamps[interp_mask],
        tongue_timestamps[keep_mask],
        processed_y[keep_mask],
    )
```

iii. The justification comes directly from the notes and paper summary: use mean fill when the tongue is occluded and a five-sigma velocity rule plus interpolation for outliers.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After preprocessing the full session trace, the agent computes the 40th and 60th percentiles of `processed_y`. Per bin, it assigns class 0 below the 40th percentile, class 1 from the 40th through the 60th percentile, and class 2 above the 60th percentile.

ii. ```python
p40, p60 = np.percentile(processed_y, [40.0, 60.0])
...
tongue_bin = np.zeros(N_BINS, dtype=np.int16)
tongue_bin[tongue_y >= tongue_info["p40"]] = 1
tongue_bin[tongue_y > tongue_info["p60"]] = 2
```

iii. The notes explicitly say the discretization should use per-session 40th and 60th percentiles to match the task definition.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The agent aligns tongue output to neural bins by taking, for each neural bin center, the closest preceding tongue-tracking frame and then discretizing that y value using the session-level thresholds.

ii. ```python
tongue_frame_idx = np.searchsorted(tongue_timestamps, bin_centers_abs, side="right") - 1
tongue_frame_idx = np.clip(tongue_frame_idx, 0, len(processed_tongue_y) - 1)
tongue_y = processed_tongue_y[tongue_frame_idx]
```

iii. The notes justify this as matching the reference marker-alignment logic more closely than averaging over wide windows.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent uses a set of local fallbacks rather than dropping every imperfect record: it parses `N/A` photostim fields as missing, defaults missing sample onset to `go - 1.85 s`, fills low-likelihood tongue frames with the session mean, interpolates tongue outliers/NaNs, clips tongue frame indices to valid bounds, infers ignore-trial choice from lick timing or instruction, and excludes sessions/trials only when neural coverage is unusable.

ii. ```python
def parse_optional_float(value: str) -> float | None:
    if value in {"N/A", "", "nan", "None"}:
        return None
```

```python
if len(sample_candidates) == 0:
    sample_onset = go_time - 1.85
...
if not np.any(good_mask):
    processed_y = np.full_like(y, session_mean_y, dtype=np.float64)
...
return CHOICE_MAP[instruction], "ignore:instruction_fallback"
```

iii. The notes frame these as “minor-mistake” handlers and document the fallback counts; notably, they report zero missing-sample-onset fallbacks on the final dataset.

## 10-a. What are the most time-consuming steps of the code?

i. The main expensive step is converting spike times into per-trial binned firing rates for all good units in each session. Session-wise HDF5 reads and tongue preprocessing are secondary costs, but the agent’s own notes identify neural binning as the key bottleneck it optimized.

ii. ```python
neural_tensor = np.empty((n_trials, n_good_units, N_BINS), dtype=np.float16)
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    counts = np.diff(edge_idx, axis=1)
    neural_tensor[:, unit_idx, :] = (counts / BIN_WIDTH_S).astype(np.float16)
```

iii. In Step 6, the agent explicitly says per-trial Python loops over neurons would be too slow and that vectorized per-unit `searchsorted` across all trials was the main speedup.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-construction loop in `process_session`, the per-session region-index loop in `build_dataset`, and the per-unit spike-binning loop are the clearest remaining vectorization targets. The code already vectorizes across trials within each unit, but not across units or trial metadata construction.

ii. ```python
for trial_idx in range(len(trial_start)):
    ...
    trial_records.append({...})
```

```python
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
```

```python
for i, label in enumerate(result.brain_region_labels):
    if label not in region_to_idx:
        region_to_idx[label] = len(brain_regions)
        brain_regions.append(label)
    session_region_idx[i] = region_to_idx[label]
```

iii. The notes acknowledge that nested per-trial/per-neuron loops would be too slow and describe the current implementation as only partially vectorized.

## 10-c. What processing does the code repeat multiple times?

i. The code reopens NWB files in `select_files` to scan for good units and then reopens them again in `process_session`; it also calls `select_files(..., sample_mode=False)` a second time at the end solely to estimate projected full-conversion time. It also duplicates per-trial constants across all output bins.

ii. ```python
session_files, excluded_sessions = select_files(all_files, sample_mode=sample_mode)
...
result = process_session(path, make_plot=make_plot)
...
est_full_s = mean_session_s * max(1, len(select_files(all_files, sample_mode=False)[0]))
```

```python
output_row[0, :] = choice_code
output_row[1, :] = outcome_code
output_row[2, :] = early_code
```

iii. The agent’s notes mention the extra selection pass indirectly in the timing estimate logic and justify the repeated output broadcasting as a simple way to satisfy the common decoder shape.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code builds `plot_payload` data and a full plotting path that are only used when `--show-processing` is enabled; it also computes detailed stats/metadata counters used for logging rather than decoding, and it expands trial-level choice/outcome/early labels across all 80 bins even though they are not intrinsically time varying.

ii. ```python
plot_payload = None
if make_plot:
    plot_payload = {
        "trial_idx": raw_trial_idx,
        ...
        "neural_trial": neural_tensor[trial_idx].astype(np.float32),
    }
```

```python
stats = {
    "n_trials": n_trials,
    "choice_sources": dict(choice_sources),
    "n_missing_sample_onset_fallback": n_missing_sample_onset_fallback,
    ...
}
```

```python
output_row[0, :] = choice_code
output_row[1, :] = outcome_code
output_row[2, :] = early_code
```

iii. The notes themselves call out the plotting path as a validation aid and the replicated outputs as a format convenience rather than something required by the underlying neuroscience analysis.
