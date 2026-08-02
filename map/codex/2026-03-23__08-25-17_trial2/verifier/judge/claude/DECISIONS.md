# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files stored in `/app/data/` using `h5py` (not PyNWB) for speed. It discovers all session files by globbing `sub-*/*.nwb` under the data directory. Each NWB file is opened once per session, and all relevant datasets (units, trials, behavioral events, behavioral time series) are read in a single `with h5py.File(...)` block. A pre-filtering step (`select_files`) opens each file first to check whether it has any classifier-good units, excluding files with zero good units.

ii.
```python
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
    if sample_mode:
        return selected[:2], excluded
    return selected, excluded
```

iii. The AI chose h5py over PyNWB for speed and lower overhead. The pre-filtering step ensures sessions without any classifier-good units are excluded before processing, consistent with the paper's 173-session count (174 NWB files minus 1 session with zero good units). This is documented in CONVERSION_NOTES.md Step 4.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is extracted from each NWB file's `general/subject/subject_id` field. Unique subject IDs are collected across sessions and stored in the `subjects` list, with each session assigned a `subject_idx` index.

ii.
```python
subject_id = str(f["general"]["subject"]["subject_id"][()])
if subject_id.startswith("b'"):
    subject_id = subject_id[2:-1]
subject_id = f"sub-{subject_id}"
```

iii. The AI uses the NWB metadata directly rather than parsing filenames. It adds a `sub-` prefix to match DANDI conventions and handles byte-string encoding artifacts. The final count of 28 subjects matches the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file constitutes one session. The session ID is derived from the file stem by stripping the behavior/ecephys/ogen suffix. Sessions with zero classifier-good units are excluded. In `--sample` mode, only the first 2 valid sessions are processed.

ii.
```python
session_id = path.stem.replace("_behavior+ecephys+ogen", "").replace(
    "_behavior+ecephys", ""
)
```

iii. The AI treats one NWB file = one session, consistent with the DANDI archive structure. 173 sessions are retained after excluding the one file with all-NaN classification labels.

## 1-d. How are the data split into trials?

i. Trials are defined by the `intervals/trials` table in each NWB file. The AI iterates over all trials in the table using `trial_start` and `trial_stop` times. For each trial, the go cue is located within the trial interval. Trials are excluded if (a) no go cue is found, (b) the full [-2.5, +1.5] neural window is not covered by the unit's observation intervals, or (c) the resulting neural matrix is all zeros.

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
        raise ValueError(...)
    go_time = float(go_candidates[-1])
```

iii. The AI uses the NWB trial table directly. The go cue for each trial is the last `go_start_times` event within the trial interval, which handles early-lick replay trials correctly (the last go cue corresponds to the final, successful trial epoch).

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three quality filters: (1) the full go-aligned neural window [-2.5, +1.5] s must fall within a good unit's observation intervals; (2) trials without a go cue event are rejected (raises an error); (3) trials with all-zero neural activity across all units and time bins are dropped. Importantly, the AI does NOT filter out early-lick trials, ignore trials, auto-water trials, free-water trials, or photostimulation trials, because these trial types contain decoder target/input variables.

ii.
```python
# Filter 1: Observation interval coverage
covered = np.any(
    (window_start >= session_obs_intervals[:, 0])
    & (window_end <= session_obs_intervals[:, 1])
)
if not covered:
    n_trials_dropped_outside_obs += 1
    continue

# Filter 3: All-zero neural trials
nonzero_trial_mask = np.any(neural_tensor != 0, axis=(1, 2))
n_trials_dropped_all_zero = int((~nonzero_trial_mask).sum())
if n_trials_dropped_all_zero:
    neural_tensor = neural_tensor[nonzero_trial_mask]
```

iii. The observation-interval check ensures valid neural data exists for the full trial window. The reference code in the papers uses stricter trial filtering (excluding early lick, auto water, free water, ignore, and stimulation trials via `get_regular_trial_mask`), but those filters are analysis-specific; the AI correctly preserves all trial types for the decoder task. The AI documents that this drops ~22% of raw trials (94,370 to 73,910).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (ragged array of absolute spike times for each unit), filtered to units where `units/classification == "good"`.

ii.
```python
classification = decode_strings(f["units"]["classification"])
good_unit_mask = classification == "good"
# ...
spike_times_flat = f["units"]["spike_times"][:]
spike_times_index = f["units"]["spike_times_index"][:]
spike_times_ragged = split_ragged(spike_times_flat, spike_times_index)
good_unit_indices = np.flatnonzero(good_unit_mask)
good_spike_times = [np.asarray(spike_times_ragged[i], dtype=np.float64) for i in good_unit_indices]
```

iii. The AI uses `classification == "good"` as the NWB-embedded equivalent of the external classifier-based QC lists described in the white paper. This yields 69,453 good units across 173 sessions, close to the paper's 69,943 (0.7% discrepancy attributed to archive version differences).

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms non-overlapping bins aligned to the go cue, spanning [-2.5, +1.5] s (80 bins total). Spike counts per bin are divided by bin width (0.05 s) to produce firing rates in Hz. The result is stored as float16 per trial.

ii.
```python
BIN_WIDTH_S = 0.05
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_WIDTH_S, dtype=np.float64)
# ...
neural_tensor = np.empty((n_trials, n_good_units, N_BINS), dtype=np.float16)
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    counts = np.diff(edge_idx, axis=1)
    neural_tensor[:, unit_idx, :] = (counts / BIN_WIDTH_S).astype(np.float16)
```

iii. The reference code uses 40 ms bins with 3.4 ms stride, but the task instructions explicitly require 50 ms bins. The AI intentionally deviates from the reference bin width to match the decoder task specification. The binning is vectorized across all trials per unit using `np.searchsorted` for efficiency.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == "good"` are included. No additional neural quality filtering (e.g., firing rate thresholds, ISI violation thresholds) is applied beyond the classifier label. The AI also drops trials where the resulting neural matrix is all zeros.

ii.
```python
good_unit_mask = classification == "good"
n_good_units = int(good_unit_mask.sum())
if n_good_units == 0:
    return None
```

iii. The classifier-based QC incorporates 15 quality metrics (ISI violation, amplitude cutoff, presence ratio, drift metric, etc.) via logistic regression trained per brain region. The AI trusts this composite label rather than reimplementing individual metric thresholds. The reference code (`helper_get_neuron_id_area`) also additionally filters by hemisphere and histology-derived CCF annotations; the AI does not replicate this additional spatial filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. For each trial, the go cue time is identified as the last `go_start_times` event within the trial's `[start_time, stop_time]` interval. Bin edges are computed relative to the go cue time: `bin_edges = go_time + BIN_EDGES_REL`.

ii.
```python
go_candidates = interval_values(go_times, start, stop)
go_time = float(go_candidates[-1])
# ...
trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL
```

iii. Go-cue alignment is consistent with the reference code and papers, which use "time to go" as the temporal reference. The use of the last go cue within each trial handles early-lick replay trials correctly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 50 ms (0.05 s), producing 80 non-overlapping bins for the [-2.5, +1.5] s window. No overlapping stride or temporal rebinning is applied; spikes are directly binned from raw spike times into the 50 ms bins.

ii.
```python
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_WIDTH_S, dtype=np.float64)
N_BINS = len(BIN_CENTERS_REL)  # 80
```

iii. The task instructions specify "50-ms-width bins for computing firing rates." The reference code uses 40 ms bins with 3.4 ms stride, but the AI correctly follows the task-specified 50 ms bin width. No rebinning from a finer resolution is needed since the AI bins directly from spike times.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This input is derived from `acquisition/BehavioralEvents/sample_start_times/timestamps` (the tone/sample onset event times) and the go cue time for temporal reference.

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

iii. The AI uses the last `sample_start_times` event before the go cue within each trial, which handles early-lick replay trials where multiple sample events may occur. The fallback of `go_time - 1.85` is provided for trials with missing sample onset events, though in practice this fallback was never triggered (documented as 0 fallbacks across the full dataset).

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each 50 ms bin, the time from tone onset is computed as: `bin_center_absolute_time - sample_onset_time`. This produces a continuous, time-varying input that increases linearly within each trial.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
```

iii. This produces a ramp-like signal starting at a negative value (before tone onset) and increasing through the trial. The result is stored as float32.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time-from-tone values are computed at the same go-cue-aligned bin centers as the neural data, ensuring perfect temporal alignment. Each bin center's value is `(go_time + bin_center_relative) - sample_onset_time`.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
input_tensor[keep_idx, 0, :] = rec["time_from_tone"]
```

iii. Since both neural and input data use the same bin centers (go_time + BIN_CENTERS_REL), temporal alignment is guaranteed by construction.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from three trial-table columns: `trials/photostim_onset` (onset time relative to trial start), `trials/photostim_duration`, and `trials/photostim_power`. All three are read as strings and parsed to floats.

ii.
```python
trial_photostim_onset = decode_strings(trials["photostim_onset"])
trial_photostim_duration = decode_strings(trials["photostim_duration"])
trial_photostim_power = decode_strings(trials["photostim_power"])
```

iii. The AI reads these fields as strings and converts to floats, handling `N/A` values gracefully. Trials with `N/A` power have no stimulation.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For trials with valid photostim parameters (non-null power, onset, and duration), the AI converts the trial-relative onset to absolute time (`trial_start + onset`), computes the stimulation window, and creates a binary array where bin centers falling within the stim window are set to 1.0, all others to 0.0.

ii.
```python
photostim_row = np.zeros(N_BINS, dtype=np.float32)
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

iii. The binary representation matches the task instruction "Whether photostimulation is on at every time point (discrete, time-varying)." The reference code aligns stimulation times to go cue; the AI instead works in absolute time and evaluates at go-aligned bin centers, which is mathematically equivalent.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostimulation binary array is evaluated at the same go-cue-aligned bin centers as the neural data. Bin centers within the stimulation window are set to 1, others to 0.

ii.
```python
photostim_row = (
    (bin_centers_abs >= stim_start_abs) & (bin_centers_abs < stim_stop_abs)
).astype(np.float32)
```

iii. Alignment is guaranteed by using the same bin center array. The reference description indicates photoinhibition occurs during the last 0.5 s of the delay epoch (ending before the go cue), so the photostim signal should appear at negative time-to-go values, consistent with the task structure.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from `trials/trial_instruction` (left/right), `trials/outcome` (hit/miss/ignore), and for ignore trials, from `acquisition/BehavioralEvents/left_lick_times` and `right_lick_times`.

ii.
```python
trial_instruction = decode_strings(trials["trial_instruction"])
trial_outcome = decode_strings(trials["outcome"])
# ...
left_lick_times = events["left_lick_times"]["timestamps"][:].astype(np.float64)
right_lick_times = events["right_lick_times"]["timestamps"][:].astype(np.float64)
```

iii. The reference code derives choice from `trial_type` (instruction) and `correctness` (outcome); the AI uses the equivalent NWB fields.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For hit trials, choice = instructed side (left=0, right=1). For miss trials, choice = opposite of instructed side. For ignore trials, the AI uses the earliest lick event (left or right) within the trial interval; if no licks are found, it falls back to the instructed side. Choice is per-trial and broadcast across all 80 time bins.

ii.
```python
def infer_choice(...) -> tuple[int, str]:
    if outcome == "hit":
        return CHOICE_MAP[instruction], "instruction+outcome"
    if outcome == "miss":
        opposite = "right" if instruction == "left" else "left"
        return CHOICE_MAP[opposite], "instruction+outcome"
    left_trial = interval_values(left_lick_times, trial_start, trial_stop)
    right_trial = interval_values(right_lick_times, trial_start, trial_stop)
    if len(left_trial) and len(right_trial):
        side = "left" if left_trial[0] <= right_trial[0] else "right"
        return CHOICE_MAP[side], "ignore:first_lick_in_trial"
    if len(left_trial):
        return CHOICE_MAP["left"], "ignore:first_lick_in_trial"
    if len(right_trial):
        return CHOICE_MAP["right"], "ignore:first_lick_in_trial"
    return CHOICE_MAP[instruction], "ignore:instruction_fallback"
```

iii. The reference code excludes ignore trials from analysis (`get_regular_trial_mask` removes `correctness == -1`), so there is no reference for handling this case. The AI's fallback strategy is a task-driven compromise documented in CONVERSION_NOTES.md. 13,258 ignore trials used the fallback, representing ~17.9% of kept trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived directly from `trials/outcome`, which contains string values "hit", "miss", or "ignore".

ii.
```python
trial_outcome = decode_strings(trials["outcome"])
# ...
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
outcome_code = OUTCOME_MAP[trial_outcome[trial_idx]]
```

iii. The mapping exactly matches the task specification: ignore=0, miss=1, hit=2.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A direct string-to-integer mapping with no additional processing. The outcome code is per-trial and broadcast across all 80 time bins.

ii.
```python
outcome_code = OUTCOME_MAP[trial_outcome[trial_idx]]
output_row[1, :] = outcome_code
```

iii. No transformation beyond the specified mapping is needed.

## 6-c. How is `output` *Outcome* aligned with the neural data?

i. Outcome is a per-trial scalar replicated across all time bins, so no temporal alignment is required beyond the trial-level correspondence.

ii.
```python
output_row = np.empty((4, N_BINS), dtype=np.int16)
output_row[1, :] = outcome_code
```

iii. The task specification states outcome is "per-trial", so broadcasting it across all time bins is the correct approach for the time-varying output format.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from `trials/early_lick`, which contains string values "early" or "no early".

ii.
```python
trial_early = decode_strings(trials["early_lick"])
# ...
EARLY_MAP = {"no early": 0, "early": 1}
early_code = EARLY_MAP[trial_early[trial_idx]]
```

iii. Direct mapping from the NWB trial table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. A direct string-to-integer mapping: "no early"=0, "early"=1. The early lick code is per-trial and broadcast across all 80 time bins.

ii.
```python
early_code = EARLY_MAP[trial_early[trial_idx]]
output_row[2, :] = early_code
```

iii. Matches the task specification exactly: no=0, yes=1.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains per-frame data with columns (x, y, likelihood) and associated timestamps at ~300 Hz.

ii.
```python
tongue_group = f["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_data = tongue_group["data"][:].astype(np.float64)
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
```

iii. The AI uses the side-view tongue tracking data, consistent with the method paper's description of side-view camera tracking of tongue, jaw, and nose movements.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI applies session-level tongue y preprocessing: (1) compute frame-to-frame velocity, (2) detect outliers exceeding 5-sigma velocity threshold, (3) replace low-likelihood frames (likelihood < 0.1) with session mean y, (4) interpolate velocity-outlier frames using linear interpolation from neighboring good frames.

ii.
```python
def process_tongue_trace(tongue_xyzl, tongue_timestamps):
    y = tongue_xyzl[:, 1].astype(np.float64, copy=True)
    likelihood = tongue_xyzl[:, 2].astype(np.float64, copy=False)
    # velocity outlier detection
    velocity = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    vel_threshold = vel_mean + 5.0 * vel_std
    # low likelihood → session mean
    processed_y[low_likelihood_mask] = session_mean_y
    # velocity outliers → interpolation
    processed_y[interp_mask] = np.interp(...)
```

iii. This matches the method paper's description: "marker outliers removed using a five-sigma velocity threshold and imputed from nearby frames" and "when the tongue is occluded in the mouth, tongue position is set to its mean value." The likelihood threshold of 0.1 is a reasonable choice for DeepLabCut tracking confidence.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Session-level 40th and 60th percentiles of the processed tongue y are computed over ALL frames in the session (including mean-filled low-likelihood frames). Values below p40 are class 0, values between p40 and p60 (inclusive) are class 1, values above p60 are class 2.

ii.
```python
p40, p60 = np.percentile(processed_y, [40.0, 60.0])
tongue_bin = np.zeros(N_BINS, dtype=np.int16)
tongue_bin[tongue_y >= tongue_info["p40"]] = 1
tongue_bin[tongue_y > tongue_info["p60"]] = 2
```

iii. The percentile computation includes mean-filled frames, which results in a heavily skewed distribution: [0.088, 0.824, 0.088] for [class 0, class 1, class 2]. This middle-class dominance occurs because many frames have tongue y at the session mean (when tongue is in the mouth/occluded), causing p40 and p60 to be close together. Despite the class imbalance, the decoder achieves 0.765 balanced accuracy for this output.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each go-aligned bin center, the AI finds the closest preceding tongue tracking frame using `np.searchsorted(..., side="right") - 1` and reads the processed tongue y value at that frame index.

ii.
```python
tongue_frame_idx = np.searchsorted(tongue_timestamps, bin_centers_abs, side="right") - 1
tongue_frame_idx = np.clip(tongue_frame_idx, 0, len(processed_tongue_y) - 1)
tongue_y = processed_tongue_y[tongue_frame_idx]
```

iii. This nearest-preceding-frame approach is similar to the reference code's `align_markers_between_lims`, which selects the last frame within each time bin window. The alignment ensures tongue y values correspond to the same time points as neural bin centers.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several missing/edge-case data scenarios:
- **Missing sample onset**: If no `sample_start_times` event is found before the go cue, falls back to `go_time - 1.85 s` (0 occurrences in practice).
- **Missing choice on ignore trials**: Uses first lick side in trial; if no licks, uses instructed side as placeholder.
- **Low-likelihood tongue frames**: Replaced with session mean tongue y.
- **Velocity-outlier tongue frames**: Linearly interpolated from neighboring good frames.
- **Byte-string encoding artifacts in subject IDs**: Stripped `b'...'` prefix.
- **Photostim fields stored as strings including "N/A"**: Parsed with `parse_optional_float` that returns None for N/A values.
- **Trials outside observation intervals**: Dropped to avoid invalid neural data.
- **All-zero neural trials**: Dropped as a final safety check.

ii.
```python
def parse_optional_float(value: str) -> float | None:
    if value in {"N/A", "", "nan", "None"}:
        return None
    return float(value)
```

iii. The AI documents fallback counts and their resolution in CONVERSION_NOTES.md Step 10. All fallback mechanisms are conservative and prefer maintaining data integrity over discarding trials.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is reading the NWB files and performing per-unit neural binning within `process_session`. Loading spike times (ragged arrays) from HDF5 and the vectorized `searchsorted` binning across all trials dominate per-session time. The AI reports ~0.70 s per session on average, with full conversion taking ~2.35 minutes for 173 sessions.

ii.
```python
spike_times_flat = f["units"]["spike_times"][:]
spike_times_index = f["units"]["spike_times_index"][:]
# ...
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    counts = np.diff(edge_idx, axis=1)
    neural_tensor[:, unit_idx, :] = (counts / BIN_WIDTH_S).astype(np.float16)
```

iii. The AI profiles timing per session and provides estimates in CONVERSION_NOTES.md Step 7.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop over trial records (lines 248-319) iterates through each trial to compute go cue, sample onset, photostim, choice, tongue y, and outputs individually. While some operations within the loop are vectorized, the overall trial-level iteration could potentially be restructured to batch-process multiple trials at once. The per-unit loop for neural binning (line 339) iterates over good units but this is already semi-vectorized (each unit processes all trials at once).

ii.
```python
for trial_idx in range(len(trial_start)):
    # ... per-trial processing ...
```

iii. The AI's approach is already reasonably efficient (~0.7 s/session). The trial loop is the main remaining Python-level iteration, but its per-iteration cost is low because the operations within it are mostly scalar or small-array operations.

## 10-c. What processing does the code repeat multiple times?

i. The `select_files` function opens every NWB file to check for good units, and then `main()` opens the same files again for full processing. Additionally, `select_files` is called twice in `main()`: once at the beginning and once inside the final statistics printing block (line 663) to estimate full-conversion time even in sample mode.

ii.
```python
session_files, excluded_sessions = select_files(all_files, sample_mode=sample_mode)
# ... processing ...
est_full_s = mean_session_s * max(1, len(select_files(all_files, sample_mode=False)[0]))
```

iii. The duplicate `select_files` call at line 663 reopens all 174 NWB files unnecessarily just to count valid sessions for a time estimate. This adds ~30 seconds of redundant I/O. The first `select_files` call is also redundant with the `process_session` function which already checks for good units (returns None if n_good_units == 0).

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The tongue y processing is performed over the entire session trace (all frames from start to end of recording), but only the values at go-aligned bin centers are actually used. The session-wide tongue trace processing (velocity computation, outlier detection, mean filling, interpolation) includes many frames far outside any trial window. Similarly, all spike times for good units are loaded into memory even though only spikes within trial windows contribute to the final binned rates. The 2D velocity computation uses both x and y coordinates of tongue tracking, but only y is ultimately used for the output.

ii.
```python
# Full session tongue processing
processed_tongue_y, tongue_info = process_tongue_trace(tongue_data, tongue_timestamps)
# But only bin-center samples are used:
tongue_frame_idx = np.searchsorted(tongue_timestamps, bin_centers_abs, side="right") - 1
tongue_y = processed_tongue_y[tongue_frame_idx]
```

iii. Processing the full trace is intentional for computing session-level percentiles and for robustness of outlier detection, but the velocity computation over inter-trial intervals is wasted work. The full spike time loading is necessary for the vectorized binning approach.
