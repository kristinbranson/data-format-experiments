# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all session files with a sorted glob over `/app/data/sub-*/*.nwb`, then opens each NWB file directly with `h5py`. Within each file it reads the `units`, `intervals/trials`, `acquisition/BehavioralEvents`, and `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking` groups.

ii.
```python
def load_candidate_files() -> list[Path]:
    return sorted(DATA_DIR.glob("sub-*/*.nwb"))
```

```python
with h5py.File(path, "r") as f:
    classification = decode_strings(f["units"]["classification"])
    ...
    trials = f["intervals"]["trials"]
    ...
    events = f["acquisition"]["BehavioralEvents"]
    ...
    tongue_group = f["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
```

iii. In `CONVERSION_NOTES.md`, the AI says it used direct HDF5 reads “for speed and lower overhead” and processed sessions one at a time to keep memory bounded.

## 1-b. How are the data split into subjects?

i. The AI reads each subject from `general/subject/subject_id`, prefixes it with `sub-`, and builds `subjects`/`subject_idx` during dataset assembly in first-seen session order.

ii.
```python
subject_id = str(f["general"]["subject"]["subject_id"][()])
if subject_id.startswith("b'"):
    subject_id = subject_id[2:-1]
subject_id = f"sub-{subject_id}"
```

```python
for result in results:
    if result.subject_id not in subject_to_idx:
        subject_to_idx[result.subject_id] = len(subjects)
        subjects.append(result.subject_id)
    subject_idx.append(subject_to_idx[result.subject_id])
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI justifies this as using the NWB `subject.subject_id` directly as the canonical subject identifier.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. It excludes files with zero `classification == "good"` units before processing, and it derives a session ID from the filename stem rather than from `nwb.identifier`.

ii.
```python
def load_candidate_files() -> list[Path]:
    return sorted(DATA_DIR.glob("sub-*/*.nwb"))
```

```python
session_id = path.stem.replace("_behavior+ecephys+ogen", "").replace(
    "_behavior+ecephys", ""
)
```

```python
for path in all_files:
    with h5py.File(path, "r") as f:
        classification = decode_strings(f["units"]["classification"])
        if np.sum(classification == "good") == 0:
            excluded.append(path.name)
            continue
    selected.append(path)
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly decided to use the 173 sessions with at least one classifier-good unit and to treat one-file-per-session as the basic session split.

## 1-d. How are the data split into trials?

i. The AI iterates over rows of `intervals/trials`, treats each row as one trial, and identifies the go cue for that trial by finding `go_start_times` that fall inside the row’s `[start_time, stop_time]` interval. If multiple go events are inside the interval, it uses the last one.

ii.
```python
trials = f["intervals"]["trials"]
trial_start = trials["start_time"][:].astype(np.float64)
trial_stop = trials["stop_time"][:].astype(np.float64)
...
for trial_idx in range(len(trial_start)):
    start = trial_start[trial_idx]
    stop = trial_stop[trial_idx]
    go_candidates = interval_values(go_times, start, stop)
    if len(go_candidates) == 0:
        raise ValueError(f"{path.name}: no go cue found for trial {trial_idx}")
    go_time = float(go_candidates[-1])
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI says it chose “event-in-trial matching rather than row order assumptions” for go-cue alignment.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out any trial whose full decoder window `[go-2.5 s, go+1.5 s]` is not fully contained inside the first good unit’s `obs_intervals`. After neural binning it drops any remaining trial with all-zero neural activity. It does not explicitly filter `free_water` trials.

ii.
```python
obs_intervals = f["units"]["obs_intervals"][:].astype(np.float64)
obs_intervals_index = f["units"]["obs_intervals_index"][:]
first_good_unit = int(good_unit_indices[0])
obs_start = 0 if first_good_unit == 0 else int(obs_intervals_index[first_good_unit - 1])
obs_stop = int(obs_intervals_index[first_good_unit])
session_obs_intervals = obs_intervals[obs_start:obs_stop]
```

```python
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
...
if n_trials_dropped_all_zero:
    neural_tensor = neural_tensor[nonzero_trial_mask]
    input_tensor = input_tensor[nonzero_trial_mask]
    output_tensor = output_tensor[nonzero_trial_mask]
```

iii. In Steps 9 and 10 of `CONVERSION_NOTES.md`, the AI says this was added after the verifier exposed long runs of all-zero trials and it wanted to exclude any trial without a complete go-aligned neural window.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `units/spike_times` for units where `units/classification == "good"`. It also uses the per-trial go-cue times to place the bin edges.

ii.
```python
classification = decode_strings(f["units"]["classification"])
good_unit_mask = classification == "good"
...
spike_times_flat = f["units"]["spike_times"][:]
spike_times_index = f["units"]["spike_times_index"][:]
spike_times_ragged = split_ragged(spike_times_flat, spike_times_index)
```

```python
go_times = events["go_start_times"]["timestamps"][:].astype(np.float64)
...
trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI maps `units.spike_times` for classifier-good units into go-aligned firing rates.

## 2-b. How is the `neural` data processed?

i. The AI reconstructs each unit’s spike train from NWB ragged arrays, bins spikes into 50 ms go-aligned windows using `np.searchsorted`, takes per-bin count differences, divides by 0.05 s to get firing rates in Hz, and stores the result as `float16`.

ii.
```python
def split_ragged(flat: np.ndarray, index: np.ndarray) -> list[np.ndarray]:
    starts = np.concatenate(([0], index[:-1]))
    return [flat[s:e] for s, e in zip(starts, index, strict=True)]
```

```python
neural_tensor = np.empty((n_trials, n_good_units, N_BINS), dtype=np.float16)
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    counts = np.diff(edge_idx, axis=1)
    neural_tensor[:, unit_idx, :] = (counts / BIN_WIDTH_S).astype(np.float16)
```

iii. In `CONVERSION_NOTES.md`, the AI says it intentionally vectorized `searchsorted` across all trials and stored neural firing rates as `float16` to reduce runtime and output size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `classification == "good"` and drops any session with zero such units.

ii.
```python
classification = decode_strings(f["units"]["classification"])
good_unit_mask = classification == "good"
n_good_units = int(good_unit_mask.sum())
if n_good_units == 0:
    return None
```

iii. In Steps 4 and 5 of `CONVERSION_NOTES.md`, the AI justifies this as the local NWB equivalent of the classifier-based QC used in the papers and reference code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns each trial to its go cue. For each trial it finds the go time inside the trial interval and then uses `go_time + BIN_EDGES_REL` to construct absolute bin edges for neural binning.

ii.
```python
go_candidates = interval_values(go_times, start, stop)
...
go_time = float(go_candidates[-1])
```

```python
trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL
...
edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
```

iii. In `CONVERSION_NOTES.md`, the AI says the reference pipeline is go-cue aligned and that this conversion keeps that alignment while changing only the final bin width/window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 50 ms bins from `-2.5 s` to `+1.5 s` relative to go cue, giving 80 bins per trial. It does not rebin from a finer neural representation; it bins spikes directly into this grid.

ii.
```python
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_WIDTH_S, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_WIDTH_S / 2.0
N_BINS = len(BIN_CENTERS_REL)
```

iii. In Steps 4 and 5 of `CONVERSION_NOTES.md`, the AI explicitly calls this a task-driven deviation from the paper’s 40 ms / 3.4 ms-stride representation.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `sample_start_times` and the trial’s go cue. For each trial it takes the last `sample_start` event between trial start and go cue. If none is found, it falls back to `go_time - 1.85`.

ii.
```python
sample_start_times = events["sample_start_times"]["timestamps"][:].astype(np.float64)
...
sample_candidates = interval_values(sample_start_times, start, go_time)
if len(sample_candidates) == 0:
    sample_onset = go_time - 1.85
    n_missing_sample_onset_fallback += 1
else:
    sample_onset = float(sample_candidates[-1])
```

iii. In Step 5 of `CONVERSION_NOTES.md` and Step 126 of the trajectory, the AI justifies the “last sample before go” rule by noting that early licks can replay the sample epoch.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes absolute bin centers for each trial and subtracts the chosen sample onset, so each bin stores seconds since tone onset. It also includes an unused fallback for missing sample-onset events.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
```

iii. In `CONVERSION_NOTES.md`, the AI says the signal should be “(bin_center_time - sample_onset_time)” for each bin. The fallback appears to be a defensive handling choice rather than a data-driven requirement.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The AI computes `time_from_tone` at the same go-aligned bin centers used for neural firing rates, so the two streams share the same 80-bin time axis.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
...
trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL
```

iii. The AI’s notes describe the decoder inputs and neural data as sharing one common go-aligned bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from trial-table fields `photostim_onset`, `photostim_duration`, and `photostim_power`, together with `trial start_time`.

ii.
```python
trial_photostim_onset = decode_strings(trials["photostim_onset"])
trial_photostim_duration = decode_strings(trials["photostim_duration"])
trial_photostim_power = decode_strings(trials["photostim_power"])
```

```python
stim_onset = parse_optional_float(trial_photostim_onset[trial_idx])
stim_dur = parse_optional_float(trial_photostim_duration[trial_idx])
stim_power = parse_optional_float(trial_photostim_power[trial_idx])
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI planned to treat trials with `photostim_power == N/A` as non-stimulated and convert trial-relative timing into the aligned bin axis.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI parses optional string-valued onset/duration/power fields, converts onset/duration to absolute start/stop times in the session clock, and marks a bin as 1 when its center falls in the stimulation interval.

ii.
```python
def parse_optional_float(value: str) -> float | None:
    if value in {"N/A", "", "nan", "None"}:
        return None
    return float(value)
```

```python
photostim_row = np.zeros(N_BINS, dtype=np.float32)
...
if stim_power is not None and stim_onset is not None and stim_dur is not None:
    stim_start_abs = start + stim_onset
    stim_stop_abs = stim_start_abs + stim_dur
    photostim_row = (
        (bin_centers_abs >= stim_start_abs) & (bin_centers_abs < stim_stop_abs)
    ).astype(np.float32)
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI says the target should be a binary time-varying series rather than a per-trial flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI aligns photostimulation by comparing absolute bin centers from the go-aligned neural grid against absolute stimulation start/stop times computed from the trial table.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
...
stim_start_abs = start + stim_onset
stim_stop_abs = stim_start_abs + stim_dur
photostim_row = (
    (bin_centers_abs >= stim_start_abs) & (bin_centers_abs < stim_stop_abs)
).astype(np.float32)
```

iii. The AI’s notes describe this as placing trial-relative photostim times onto the same go-aligned axis as the neural bins.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from `trial_instruction` and `outcome` on hit/miss trials, but for `ignore` trials it additionally consults `left_lick_times` and `right_lick_times`, then falls back to the instructed side if no lick occurs.

ii.
```python
def infer_choice(
    instruction: str,
    outcome: str,
    trial_start: float,
    trial_stop: float,
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
) -> tuple[int, str]:
    if outcome == "hit":
        return CHOICE_MAP[instruction], "instruction+outcome"
    if outcome == "miss":
        opposite = "right" if instruction == "left" else "left"
        return CHOICE_MAP[opposite], "instruction+outcome"
    ...
    return CHOICE_MAP[instruction], "ignore:instruction_fallback"
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI states that ignore trials do not carry an explicit choice label and documents this fallback as a task-driven compromise. Step 130 of the trajectory says it checked sampled sessions for post-go licks on ignore trials before deciding how to encode them.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes only two choice classes, `left=0` and `right=1`. It uses instructed side on hits, opposite side on misses, and inferred/fallback left-or-right labels on ignore trials, then repeats the chosen code across all 80 bins.

ii.
```python
CHOICE_MAP = {"left": 0, "right": 1}
```

```python
choice_code, choice_source = infer_choice(...)
...
output_row = np.empty((4, N_BINS), dtype=np.int16)
output_row[0, :] = choice_code
```

```python
"output_values": [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["<40th_pct", "40th_to_60th_pct", ">60th_pct"],
],
```

iii. In Step 5 and Step 12 of `CONVERSION_NOTES.md`, the AI justifies this as necessary because the decoder format requires categorical labels and ignore trials lack explicit ground-truth choice.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI derives outcome directly from the trials-table `outcome` column.

ii.
```python
trial_outcome = decode_strings(trials["outcome"])
...
outcome_code = OUTCOME_MAP[trial_outcome[trial_idx]]
```

iii. The AI’s notes describe `trials.outcome` as already containing the required categories `hit`, `miss`, and `ignore`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore=0`, `miss=1`, `hit=2` and repeats the per-trial label across all 80 bins.

ii.
```python
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
...
output_row[1, :] = outcome_code
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI says this mapping matches the task specification exactly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The AI derives early lick directly from the trials-table `early_lick` column.

ii.
```python
trial_early = decode_strings(trials["early_lick"])
...
early_code = EARLY_MAP[trial_early[trial_idx]]
```

iii. In `CONVERSION_NOTES.md`, the AI identifies `trials.early_lick` as a direct source variable for this decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early=0` and `early=1`, then repeats that trial label across all 80 bins.

ii.
```python
EARLY_MAP = {"no early": 0, "early": 1}
...
output_row[2, :] = early_code
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI says this mapping matches the user’s requested output coding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue y-position from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, using column 1 of `data` as y-position, column 2 as tracking likelihood, and the accompanying timestamps.

ii.
```python
tongue_group = f["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_data = tongue_group["data"][:].astype(np.float64)
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
```

```python
xy = tongue_xyzl[:, :2].astype(np.float64, copy=False)
y = tongue_xyzl[:, 1].astype(np.float64, copy=True)
likelihood = tongue_xyzl[:, 2].astype(np.float64, copy=False)
```

iii. In `CONVERSION_NOTES.md`, the AI says the local NWB side-view tongue tracking stream is the needed source for this output.

## 8-b. How is `output` *Tongue y-position* processed?

i. The AI first applies a 5-sigma frame-to-frame velocity outlier rule and a likelihood threshold of 0.1. It replaces low-likelihood frames with the session mean tongue y and linearly interpolates outliers or NaNs. It then computes session-level 40th/60th percentiles from the fully processed framewise y values. For each trial, it samples the processed tongue trace at each neural bin center using the most recent camera frame.

ii.
```python
LIKELIHOOD_THRESHOLD = 0.1
...
velocity = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
...
outlier_mask[1:] = np.isfinite(velocity) & (velocity > vel_threshold)
low_likelihood_mask = likelihood < LIKELIHOOD_THRESHOLD
```

```python
processed_y[low_likelihood_mask] = session_mean_y
interp_mask = outlier_mask | ~np.isfinite(processed_y)
...
processed_y[interp_mask] = np.interp(
    tongue_timestamps[interp_mask],
    tongue_timestamps[keep_mask],
    processed_y[keep_mask],
)
```

```python
p40, p60 = np.percentile(processed_y, [40.0, 60.0])
...
tongue_frame_idx = np.searchsorted(tongue_timestamps, bin_centers_abs, side="right") - 1
tongue_frame_idx = np.clip(tongue_frame_idx, 0, len(processed_tongue_y) - 1)
tongue_y = processed_tongue_y[tongue_frame_idx]
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly cites the method-paper description of low-likelihood mean fill and five-sigma velocity outlier handling as its rationale for this tongue preprocessing.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses two session-wide percentile thresholds, `p40` and `p60`, computed from processed framewise tongue y. It assigns category 0 below `p40`, category 1 from `p40` through `p60`, and category 2 above `p60`. It does not create a fourth “not visible” category.

ii.
```python
p40, p60 = np.percentile(processed_y, [40.0, 60.0])
```

```python
tongue_bin = np.zeros(N_BINS, dtype=np.int16)
tongue_bin[tongue_y >= tongue_info["p40"]] = 1
tongue_bin[tongue_y > tongue_info["p60"]] = 2
```

```python
"output_values": [
    ["left", "right"],
    ["ignore", "miss", "hit"],
    ["no", "yes"],
    ["<40th_pct", "40th_to_60th_pct", ">60th_pct"],
],
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI planned percentile-based discretization and described middle-bin dominance as acceptable in its validation notes.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue data by evaluating the processed continuous tongue trace at the neural bin centers. For each bin center, it uses the latest camera frame at or before that time.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
...
tongue_frame_idx = np.searchsorted(tongue_timestamps, bin_centers_abs, side="right") - 1
tongue_frame_idx = np.clip(tongue_frame_idx, 0, len(processed_tongue_y) - 1)
tongue_y = processed_tongue_y[tongue_frame_idx]
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI says this “closest preceding frame” rule was chosen because it seemed closer to the reference marker-alignment logic than averaging over long windows.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mixes exclusion and imputation. It drops sessions with no classifier-good units, drops trials whose full neural window is outside `obs_intervals`, and drops all-zero neural trials after binning. It handles missing sample onsets with a `go - 1.85 s` fallback. It fills low-likelihood tongue frames with the session mean, linearly interpolates tongue outliers/NaNs, and assigns fallback choice labels on ignore trials.

ii.
```python
if n_good_units == 0:
    return None
```

```python
if len(sample_candidates) == 0:
    sample_onset = go_time - 1.85
    n_missing_sample_onset_fallback += 1
```

```python
processed_y[low_likelihood_mask] = session_mean_y
...
processed_y[interp_mask] = np.interp(...)
```

```python
if n_trials_dropped_all_zero:
    neural_tensor = neural_tensor[nonzero_trial_mask]
    ...
```

```python
return CHOICE_MAP[instruction], "ignore:instruction_fallback"
```

iii. The AI’s notes justify these as pragmatic fixes: obs-interval filtering was added after verifier failures, the tongue imputation follows the method-paper text in its reading, and ignore-choice fallback was treated as an unavoidable placeholder for a required decoder label.

## 10-a. What are the most time-consuming steps of the code?

i. The AI’s code is dominated by per-session HDF5 reads of large ragged spike-time arrays and tongue-tracking arrays, plus the per-unit `np.searchsorted` neural binning loop. Optional plotting also adds work when enabled.

ii.
```python
spike_times_flat = f["units"]["spike_times"][:]
spike_times_index = f["units"]["spike_times_index"][:]
...
tongue_data = tongue_group["data"][:].astype(np.float64)
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
```

```python
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    counts = np.diff(edge_idx, axis=1)
    neural_tensor[:, unit_idx, :] = (counts / BIN_WIDTH_S).astype(np.float16)
```

iii. In Step 6 of `CONVERSION_NOTES.md`, the AI repeatedly frames direct HDF5 access, vectorized binning, and reduced precision as runtime/memory optimizations for the expensive neural path.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The code still uses Python loops over trials to build `trial_records` and over units to bin spikes. It also loops over sessions and over region labels during dataset assembly. The per-unit loop is partly vectorized across trials already, but the trial-building loop could be reduced further.

ii.
```python
for trial_idx in range(len(trial_start)):
    ...
    trial_records.append(...)
```

```python
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    ...
```

```python
for result in results:
    ...
    for i, label in enumerate(result.brain_region_labels):
        ...
```

iii. In `CONVERSION_NOTES.md`, the AI focuses on the neural binning loop as the main place where vectorization mattered and says it already vectorized across trials while leaving some Python-level loops in place.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats some work that the reference solution avoids. It first accumulates per-trial dictionaries in `trial_records` and then copies those values into dense tensors in a second pass. It also calls `select_files` twice in `main`, and it reconstructs ragged spike times for all units before subselecting the good ones.

ii.
```python
trial_records.append(
    {
        "raw_trial_idx": trial_idx,
        "go_time": go_time,
        "sample_onset": sample_onset,
        "time_from_tone": time_from_tone,
        "photostim_on": photostim_row,
        "output": output_row,
    }
)
...
for keep_idx, rec in enumerate(trial_records):
    go_per_trial[keep_idx] = rec["go_time"]
    ...
    output_tensor[keep_idx] = rec["output"]
```

```python
all_files = load_candidate_files()
session_files, excluded_sessions = select_files(all_files, sample_mode=sample_mode)
...
est_full_s = mean_session_s * max(1, len(select_files(all_files, sample_mode=False)[0]))
```

```python
spike_times_ragged = split_ragged(spike_times_flat, spike_times_index)
good_unit_indices = np.flatnonzero(good_unit_mask)
good_spike_times = [np.asarray(spike_times_ragged[i], dtype=np.float64) for i in good_unit_indices]
```

iii. The AI does not call these out explicitly in its notes; this is mostly evident from the structure of `convert_data.py`.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes extra bookkeeping and plotting support that are not needed by downstream decoding on `converted_data.pkl`: `plot_payload`, multiple per-session stats counters, processing figures, session timing estimates, and some intermediate arrays used only for plots/stats. It also reads `photostim_power` even though onset/duration already indicate non-stim trials.

ii.
```python
plot_payload = None
if make_plot:
    ...
    plot_payload = {
        "trial_idx": raw_trial_idx,
        ...
        "neural_trial": neural_tensor[trial_idx].astype(np.float32),
        "choice": int(output_tensor[trial_idx, 0, 0]),
        "outcome": int(output_tensor[trial_idx, 1, 0]),
        "early": int(output_tensor[trial_idx, 2, 0]),
    }
```

```python
stats = {
    "n_trials": n_trials,
    "n_good_units": n_good_units,
    "choice_sources": dict(choice_sources),
    "n_missing_sample_onset_fallback": n_missing_sample_onset_fallback,
    "n_trials_dropped_outside_obs": n_trials_dropped_outside_obs,
    "n_trials_dropped_all_zero": n_trials_dropped_all_zero,
    "tongue_info": tongue_info,
    "stim_trials": int(np.sum(input_tensor[:, 1, :].any(axis=1))),
    "session_seconds": float(time.time() - session_start),
}
```

```python
stim_power = parse_optional_float(trial_photostim_power[trial_idx])
if stim_power is not None and stim_onset is not None and stim_dur is not None:
```

iii. The AI’s notes justify these extras as validation, diagnostics, and runtime-estimation support rather than as part of the core converted dataset.
