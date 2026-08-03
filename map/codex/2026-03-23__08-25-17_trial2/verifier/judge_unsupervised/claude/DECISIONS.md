# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB files using `h5py`. It globs all `.nwb` files under `data/sub-*/*.nwb`, then opens each file to read units (spike times, classification, anno_name, obs_intervals), trial intervals (start/stop time, instruction, outcome, early_lick, photostim fields), behavioral events (go cue, sample start, left/right lick times), and behavioral time series (tongue tracking). All sessions are processed sequentially in a single loop, with results accumulated into lists.

ii.
```python
def load_candidate_files() -> list[Path]:
    return sorted(DATA_DIR.glob("sub-*/*.nwb"))

def process_session(path: Path, make_plot: bool = False) -> SessionResult | None:
    with h5py.File(path, "r") as f:
        classification = decode_strings(f["units"]["classification"])
        good_unit_mask = classification == "good"
        # ... reads spike_times, trials, events, tongue tracking ...
```

iii. The AI chose h5py over PyNWB for speed. From CONVERSION_NOTES Step 6: "Loads NWB directly with h5py instead of PyNWB for speed and lower overhead."

## 1-b. How are the data split into subjects (mice)?

i. Subject IDs are extracted from each NWB file's `general/subject/subject_id` field. Unique subject IDs are accumulated across sessions, and each session is assigned a `subject_idx` pointing into the unique subjects list.

ii.
```python
subject_id = str(f["general"]["subject"]["subject_id"][()])
if subject_id.startswith("b'"):
    subject_id = subject_id[2:-1]
subject_id = f"sub-{subject_id}"
```

iii. The AI prefixes subject IDs with "sub-" to match the DANDI/NWB folder naming convention. 28 unique subjects were found, matching the paper's reported count.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The AI excludes files where no units have `classification == "good"`, resulting in 173 sessions (from 174 NWB files). One session (`sub-440958_ses-20190216T162508`) was excluded because all its units had `classification == nan`.

ii.
```python
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

iii. From CONVERSION_NOTES Step 4: "Exclude the single session ... which has classification == nan for all 1,852 units and zero classifier-labeled good units. This reconciles the session count to 173."

## 1-d. How are the data split into trials?

i. Trials are read from the `intervals/trials` table in each NWB file. Each row is a trial with `start_time` and `stop_time`. Go cue events are matched to trials by finding go cue timestamps within each trial's time window.

ii.
```python
trials = f["intervals"]["trials"]
trial_start = trials["start_time"][:].astype(np.float64)
trial_stop = trials["stop_time"][:].astype(np.float64)
# ...
for trial_idx in range(len(trial_start)):
    start = trial_start[trial_idx]
    stop = trial_stop[trial_idx]
    go_candidates = interval_values(go_times, start, stop)
    if len(go_candidates) == 0:
        raise ValueError(f"{path.name}: no go cue found for trial {trial_idx}")
    go_time = float(go_candidates[-1])
```

iii. The AI uses the last go cue event within each trial interval as the alignment point, reasoning that early-lick replay trials may contain multiple go cue events.

## 1-e. How are trials filtered based on quality controls?

i. Two filters are applied: (1) Trials are excluded if the full [-2.5, +1.5]s window around the go cue is not covered by the observation intervals of the first good unit. (2) Trials with all-zero neural activity across all units and time bins are dropped. No filtering based on behavioral criteria (auto_water, free_water, early_lick, outcome) is applied. Session-level behavioral performance criteria (>65% correct, >=50 correct trials per direction) are NOT applied.

ii.
```python
# Obs-interval coverage filter
covered = np.any(
    (window_start >= session_obs_intervals[:, 0])
    & (window_end <= session_obs_intervals[:, 1])
)
if not covered:
    n_trials_dropped_outside_obs += 1
    continue

# All-zero neural trial filter (after binning)
nonzero_trial_mask = np.any(neural_tensor != 0, axis=(1, 2))
```

iii. From CONVERSION_NOTES Step 4: "Treat these [session-level criteria] as analysis-specific selection criteria for paper figures, not as the base dataset definition." The obs_intervals filter uses only the first good unit's intervals (lines 217-221 of convert_data.py), not the intersection of all units' intervals.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` for units where `units/classification == "good"`.

ii.
```python
spike_times_flat = f["units"]["spike_times"][:]
spike_times_index = f["units"]["spike_times_index"][:]
spike_times_ragged = split_ragged(spike_times_flat, spike_times_index)
good_unit_indices = np.flatnonzero(good_unit_mask)
good_spike_times = [np.asarray(spike_times_ragged[i], dtype=np.float64) for i in good_unit_indices]
```

iii. From CONVERSION_NOTES Step 5: "Use units.classification == 'good' rather than units.unit_quality == 'good'. This is the only local field consistent with the classifier-based QC described in the white paper and methods."

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins spanning [-2.5, +1.5]s relative to go cue (80 bins total). Spike counts are converted to firing rates by dividing by the bin width (0.05s). Results are stored as float16.

ii.
```python
BIN_WIDTH_S = 0.05
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_WIDTH_S, dtype=np.float64)

# Vectorized binning per unit across all trials
neural_tensor = np.empty((n_trials, n_good_units, N_BINS), dtype=np.float16)
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    counts = np.diff(edge_idx, axis=1)
    neural_tensor[:, unit_idx, :] = (counts / BIN_WIDTH_S).astype(np.float16)
```

iii. From CONVERSION_NOTES Step 5: "Store firing rates, not spike counts, because the task explicitly asks for 50 ms bins 'for computing firing rates'." The reference code uses 40ms bins with 3.4ms stride, but 50ms non-overlapping bins are used per the decoder task instructions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by `units.classification == "good"`. This yielded 69,453 good units across 173 sessions (vs. 69,943 reported in the paper, a 0.7% difference). No additional QC metrics (presence_ratio, ISI violation, drift, etc.) are applied beyond the classification field.

ii.
```python
classification = decode_strings(f["units"]["classification"])
good_unit_mask = classification == "good"
n_good_units = int(good_unit_mask.sum())
if n_good_units == 0:
    return None
```

iii. From CONVERSION_NOTES Step 4: "Use units.classification == 'good' as the local equivalent of the external QC lists. unit_quality == 'good' is too permissive and produces 154,948 units, which is far above the published total."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go cue onset. For each trial, the go cue is found as the last go_start_times event within the trial's [start_time, stop_time] interval. Bin edges are computed as go_time + BIN_EDGES_REL.

ii.
```python
go_candidates = interval_values(go_times, start, stop)
go_time = float(go_candidates[-1])
# ...
trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL
```

iii. The instructions specify "Temporally align based on Go cue onset." The AI identified the go cue from `acquisition/BehavioralEvents/go_start_times/timestamps`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms bins, as specified in the decoder task instructions. No temporal rebinning is applied — spike times are directly binned at this resolution. The reference paper uses 40ms bins with 3.4ms stride, but the instructions explicitly require 50ms bins.

ii.
```python
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_WIDTH_S, dtype=np.float64)
N_BINS = len(BIN_CENTERS_REL)  # = 80
```

iii. From CONVERSION_NOTES Step 5: "Keep reference curation and go-cue temporal alignment, but intentionally change only the final bin width/window to 50 ms and [-2.5, +1.5] s because the decoder task requires it."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `acquisition/BehavioralEvents/sample_start_times/timestamps` and `acquisition/BehavioralEvents/go_start_times/timestamps`.

ii.
```python
sample_start_times = events["sample_start_times"]["timestamps"][:].astype(np.float64)
# ...
sample_candidates = interval_values(sample_start_times, start, go_time)
if len(sample_candidates) == 0:
    sample_onset = go_time - 1.85
    n_missing_sample_onset_fallback += 1
else:
    sample_onset = float(sample_candidates[-1])
```

iii. The AI uses `sample_start_times` as the tone onset event. From CONVERSION_NOTES Step 5: "For each trial, use the last sample-start event before the go cue as the effective tone/sample onset."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the last `sample_start` event occurring between trial start and go cue is identified as the tone onset. Time from tone onset is computed as the absolute bin center time minus the sample onset time. If no sample start event is found, a fallback of `go_time - 1.85s` is used.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
```

iii. The fallback of 1.85s before the go cue was chosen based on task timing (0.65s sample + 1.2s delay = 1.85s). In practice, the fallback was never needed (n_missing_sample_onset_fallback = 0).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Time from tone onset is computed at the same bin centers as the neural data (go_time + BIN_CENTERS_REL), so they are inherently aligned.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
# stored in input_tensor[keep_idx, 0, :]
```

iii. Both neural and input data use the same 80 bin centers relative to go cue.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `intervals/trials` columns: `photostim_onset`, `photostim_duration`, `photostim_power`, and `start_time`.

ii.
```python
trial_photostim_onset = decode_strings(trials["photostim_onset"])
trial_photostim_duration = decode_strings(trials["photostim_duration"])
trial_photostim_power = decode_strings(trials["photostim_power"])
```

iii. These are per-trial fields from the NWB trial table storing photostimulation parameters.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. If photostim_power, onset, and duration are all non-N/A, the stimulation start is computed as trial_start_time + photostim_onset, and the stop as start + duration. A binary indicator is created at each bin center: 1 if the bin center falls within [stim_start, stim_stop), 0 otherwise. If any parameter is N/A, the entire trial gets all zeros.

ii.
```python
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

iii. From CONVERSION_NOTES Step 5: "Convert trial-relative onset/duration to absolute time, then to go-aligned binary series over bins."

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is evaluated at the same bin centers as neural data (go_time + BIN_CENTERS_REL), ensuring alignment.

ii.
```python
photostim_row = (
    (bin_centers_abs >= stim_start_abs) & (bin_centers_abs < stim_stop_abs)
).astype(np.float32)
# bin_centers_abs = go_time + BIN_CENTERS_REL, same as neural
```

iii. Same temporal grid as all other data streams.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trials/trial_instruction`, `trials/outcome`, and `acquisition/BehavioralEvents/left_lick_times` and `right_lick_times` (for ignore trials).

ii.
```python
trial_instruction = decode_strings(trials["trial_instruction"])
trial_outcome = decode_strings(trials["outcome"])
left_lick_times = events["left_lick_times"]["timestamps"][:].astype(np.float64)
right_lick_times = events["right_lick_times"]["timestamps"][:].astype(np.float64)
```

iii. Choice is not directly stored but inferred from instruction and outcome.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For hit trials, choice = instructed side (left=0, right=1). For miss trials, choice = opposite of instructed side. For ignore trials, choice is determined by the first lick within the trial interval (using left_lick_times and right_lick_times); if no licks exist, choice defaults to the instructed side as a fallback. The choice value is then repeated across all 80 time bins.

ii.
```python
def infer_choice(instruction, outcome, trial_start, trial_stop, left_lick_times, right_lick_times):
    if outcome == "hit":
        return CHOICE_MAP[instruction], "instruction+outcome"
    if outcome == "miss":
        opposite = "right" if instruction == "left" else "left"
        return CHOICE_MAP[opposite], "instruction+outcome"
    # ignore trials: use first lick in trial
    left_trial = interval_values(left_lick_times, trial_start, trial_stop)
    right_trial = interval_values(right_lick_times, trial_start, trial_stop)
    if len(left_trial) and len(right_trial):
        side = "left" if left_trial[0] <= right_trial[0] else "right"
        return CHOICE_MAP[side], "ignore:first_lick_in_trial"
    # ...
    return CHOICE_MAP[instruction], "ignore:instruction_fallback"
```

iii. From CONVERSION_NOTES Step 5: "Because the source does not contain an explicit choice label when no response occurs, use the earliest lick side in the trial if available; otherwise fall back to instructed side."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived directly from `intervals/trials/outcome`.

ii.
```python
trial_outcome = decode_strings(trials["outcome"])
outcome_code = OUTCOME_MAP[trial_outcome[trial_idx]]
```

iii. Direct mapping from the NWB trial table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String values are mapped to integers: ignore=0, miss=1, hit=2. The outcome value is replicated across all 80 time bins.

ii.
```python
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
outcome_code = OUTCOME_MAP[trial_outcome[trial_idx]]
output_row[1, :] = outcome_code
```

iii. Matches the instruction specification exactly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived directly from `intervals/trials/early_lick`.

ii.
```python
trial_early = decode_strings(trials["early_lick"])
early_code = EARLY_MAP[trial_early[trial_idx]]
```

iii. Direct mapping from the NWB trial table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. String values are mapped to integers: "no early"=0, "early"=1. The value is replicated across all 80 time bins.

ii.
```python
EARLY_MAP = {"no early": 0, "early": 1}
early_code = EARLY_MAP[trial_early[trial_idx]]
output_row[2, :] = early_code
```

iii. Matches the instruction specification exactly.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` (columns: x, y, likelihood) and its associated `timestamps`.

ii.
```python
tongue_group = f["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_data = tongue_group["data"][:].astype(np.float64)  # shape (N, 3): x, y, likelihood
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
```

iii. The side-view camera tongue tracking data from the NWB file.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Session-wide tongue y is processed with: (1) velocity-based outlier detection at 5-sigma threshold, (2) low-likelihood frames (likelihood < 0.1) are replaced with the session mean y, (3) outlier frames are linearly interpolated from neighboring good values. Session-level 40th and 60th percentiles are computed on the full processed trace (including imputed values).

ii.
```python
def process_tongue_trace(tongue_xyzl, tongue_timestamps):
    y = tongue_xyzl[:, 1].astype(np.float64, copy=True)
    likelihood = tongue_xyzl[:, 2].astype(np.float64, copy=False)
    # velocity outlier detection
    dt = np.diff(tongue_timestamps)
    velocity = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    vel_threshold = vel_mean + 5.0 * vel_std
    # low likelihood -> session mean
    processed_y[low_likelihood_mask] = session_mean_y
    # outliers -> interpolation
    processed_y[interp_mask] = np.interp(...)
    p40, p60 = np.percentile(processed_y, [40.0, 60.0])
```

iii. From CONVERSION_NOTES Step 5: "Planned preprocessing: 5-sigma velocity outlier detection + interpolation; low-likelihood/occluded frames imputed to session mean tongue y." The method paper describes removing marker outliers using a five-sigma velocity threshold and setting occluded tongue positions to their mean value.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles of the processed tongue y trace are computed. At each bin center: class 0 if y < p40, class 1 if p40 <= y <= p60, class 2 if y > p60. Percentiles are computed over the entire session's processed trace, including frames imputed to the session mean.

ii.
```python
p40, p60 = np.percentile(processed_y, [40.0, 60.0])
# ...
tongue_bin = np.zeros(N_BINS, dtype=np.int16)
tongue_bin[tongue_y >= tongue_info["p40"]] = 1
tongue_bin[tongue_y > tongue_info["p60"]] = 2
```

iii. This produces a distribution of approximately [0.088, 0.824, 0.088] across the full dataset, heavily dominated by the middle class. This is because a large fraction of tongue frames have low likelihood (tongue in mouth) and are set to the session mean, which clusters near the 40th-60th percentile range.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. At each of the 80 bin centers (aligned to go cue), the tongue y value is looked up from the processed continuous trace using the closest preceding frame (nearest-neighbor lookup via `searchsorted`).

ii.
```python
tongue_frame_idx = np.searchsorted(tongue_timestamps, bin_centers_abs, side="right") - 1
tongue_frame_idx = np.clip(tongue_frame_idx, 0, len(processed_tongue_y) - 1)
tongue_y = processed_tongue_y[tongue_frame_idx]
```

iii. From CONVERSION_NOTES Step 5: "For each 50 ms neural bin, assign tongue y from the processed continuous tongue trace at the bin center (or closest preceding frame)."

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled: (1) Sessions with zero good units are excluded entirely. (2) Trials without a go cue raise an error. (3) Missing sample onset events use a fallback of go_time - 1.85s (never triggered). (4) N/A photostim parameters result in all-zero photostim input. (5) Ignore trials with no licks use instructed side as choice fallback. (6) Low-likelihood tongue frames are imputed to session mean. (7) Tongue velocity outliers are linearly interpolated. (8) Trials outside obs_intervals are dropped. (9) All-zero neural trials are dropped. (10) Subject IDs with byte-string artifacts are cleaned.

ii.
```python
# Example: subject ID cleaning
if subject_id.startswith("b'"):
    subject_id = subject_id[2:-1]

# Example: missing sample onset fallback
if len(sample_candidates) == 0:
    sample_onset = go_time - 1.85
    n_missing_sample_onset_fallback += 1
```

iii. From CONVERSION_NOTES Step 10: "No new mismatches were found in Step 10. The existing observation-interval fix remained sufficient, and the raw-to-converted sanity checks passed without additional code changes."

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading NWB data from disk via h5py (spike_times, tongue tracking data). (2) Per-unit spike binning across all trials using `np.searchsorted`. The full conversion ran at approximately 0.7-1.5 seconds per session for 173 sessions, completing in about 3 minutes.

ii.
```python
# Per-unit binning loop - main computational bottleneck
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    counts = np.diff(edge_idx, axis=1)
    neural_tensor[:, unit_idx, :] = (counts / BIN_WIDTH_S).astype(np.float16)
```

iii. From conversion_full_out.txt, per-session times ranged from ~0.5s to ~1.7s. The AI optimized with vectorized binning but noted efficiency improvements were possible.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop (lines 248-319) iterates over each trial to find go cues, compute sample onset, construct photostim indicators, infer choice, and look up tongue position. Many of these operations (go cue matching, sample onset finding, photostim interval checking) could be vectorized across trials.

ii.
```python
for trial_idx in range(len(trial_start)):
    start = trial_start[trial_idx]
    stop = trial_stop[trial_idx]
    go_candidates = interval_values(go_times, start, stop)
    # ... per-trial processing ...
```

iii. The AI vectorized the neural binning step (per-unit across all trials) but left the trial metadata extraction as a Python loop. This is the main remaining vectorization opportunity.

## 10-c. What processing does the code repeat multiple times?

i. The `select_files` function is called twice: once in `main()` to select files, and again at line 663 to estimate full conversion time. This re-opens all NWB files a second time to check classification counts, which is redundant I/O.

ii.
```python
# Called once in main()
session_files, excluded_sessions = select_files(all_files, sample_mode=sample_mode)

# Called again at line 663 for time estimation
est_full_s = mean_session_s * max(1, len(select_files(all_files, sample_mode=False)[0]))
```

iii. This redundant call is minor in terms of total runtime but is unnecessary repeated I/O.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The tongue trace is processed for the entire session duration, but only the portions overlapping with trial windows are used. (2) Trials that end up being all-zero neural trials are fully processed (input/output construction, tongue lookup) before being discarded. (3) The velocity computation and outlier detection process the full tongue trace including periods between trials that are never sampled.

ii.
```python
# Full session tongue processing, but only trial-aligned portions used
processed_tongue_y, tongue_info = process_tongue_trace(tongue_data, tongue_timestamps)

# Trials fully processed then dropped if all-zero
nonzero_trial_mask = np.any(neural_tensor != 0, axis=(1, 2))
if n_trials_dropped_all_zero:
    neural_tensor = neural_tensor[nonzero_trial_mask]
    input_tensor = input_tensor[nonzero_trial_mask]
    output_tensor = output_tensor[nonzero_trial_mask]
```

iii. The full-session tongue processing is intentional since session-level percentiles are needed for discretization, though the velocity outlier detection on inter-trial periods is unnecessary for the final output.
