# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files under `/app/data/sub-*/`. It uses `h5py` directly (not `pynwb`) for speed, globbing all `.nwb` files with `sorted(DATA_DIR.glob("sub-*/*.nwb"))`. Before processing, it pre-scans every file to check for good units and excludes sessions with zero classifier-good units. Each remaining file is opened with `h5py.File(path, "r")` and the relevant groups (`units`, `intervals/trials`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`) are read. Subjects, trials, and units are extracted from within each file.

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
    ...

def process_session(path: Path, make_plot: bool = False) -> SessionResult | None:
    with h5py.File(path, "r") as f:
        classification = decode_strings(f["units"]["classification"])
        ...
```

iii. The AI chose `h5py` over `pynwb` explicitly for speed and lower overhead (documented in CONVERSION_NOTES.md Step 6). The glob pattern matches the DANDI dataset layout, one NWB file per session. The pre-scan for good units avoids processing sessions that will be dropped anyway.

## 1-b. How are the data split into subjects?

i. The AI reads `subject_id` from `general/subject/subject_id` in each NWB file and prepends `"sub-"` to create subject identifiers like `"sub-440956"`. Unique subjects are collected in encounter order during assembly, with a mapping from subject_id to index.

ii.
```python
subject_id = str(f["general"]["subject"]["subject_id"][()])
if subject_id.startswith("b'"):
    subject_id = subject_id[2:-1]
subject_id = f"sub-{subject_id}"
```

Assembly:
```python
for result in results:
    if result.subject_id not in subject_to_idx:
        subject_to_idx[result.subject_id] = len(subjects)
        subjects.append(result.subject_id)
    subject_idx.append(subject_to_idx[result.subject_id])
```

iii. The AI reads the canonical NWB subject field. The `"sub-"` prefix is added to match the DANDI folder naming convention. Subject ordering follows encounter order (sorted file list), not sorted by subject ID.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session. The session ID is derived from the filename by stripping the `_behavior+ecephys+ogen` / `_behavior+ecephys` suffix from the stem, rather than using `nwb.identifier`. Sessions with zero classifier-good units are excluded in the pre-scan, yielding 173 sessions.

ii.
```python
session_id = path.stem.replace("_behavior+ecephys+ogen", "").replace(
    "_behavior+ecephys", ""
)
```

iii. The file boundary defines the session boundary. The filename-derived session ID is cosmetic; the actual session identity is determined by which file is being processed. 173 of 174 files survive after excluding the one with no good units.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`). Each row is one behavioral trial. The go cue for each trial is found by searching `go_start_times` timestamps within the trial's `[start_time, stop_time]` interval, using the last matching go cue event.

ii.
```python
trial_start = trials["start_time"][:].astype(np.float64)
trial_stop = trials["stop_time"][:].astype(np.float64)
...
go_candidates = interval_values(go_times, start, stop)
if len(go_candidates) == 0:
    raise ValueError(...)
go_time = float(go_candidates[-1])
```

iii. The AI uses `interval_values` (a searchsorted-based function) to find go cues within the trial window, taking the last one. This differs from the reference which directly indexes go cues by position (one per trial).

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two ways: (1) by observation interval coverage -- a trial is kept only if its full `[-2.5, 1.5]` go-aligned window is contained within a single obs_interval entry; (2) by all-zero neural data -- after binning, trials where all neural activity is zero across all units and time bins are dropped. Free-water trials are NOT explicitly filtered.

ii.
```python
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
n_trials_dropped_all_zero = int((~nonzero_trial_mask).sum())
if n_trials_dropped_all_zero:
    neural_tensor = neural_tensor[nonzero_trial_mask]
    ...
```

iii. The AI's obs_intervals check requires the entire analysis window to be covered, which is more conservative than the reference approach (which matches trial start times to obs_interval start times). The AI does not explicitly filter `free_water` trials. The all-zero neural check is an additional safeguard. The AI ends up with 73,910 trials vs the reference's 90,860 -- a significant difference likely caused by the stricter obs_intervals coverage requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (the flat spike times array) and `units/spike_times_index` (the ragged index), filtered to units where `classification == 'good'`. The go cue times from `BehavioralEvents/go_start_times` are used to define the bin edges.

ii.
```python
spike_times_flat = f["units"]["spike_times"][:]
spike_times_index = f["units"]["spike_times_index"][:]
spike_times_ragged = split_ragged(spike_times_flat, spike_times_index)
good_unit_indices = np.flatnonzero(good_unit_mask)
good_spike_times = [np.asarray(spike_times_ragged[i], dtype=np.float64) for i in good_unit_indices]
```

iii. `spike_times` is the only neural signal in the NWB files. The AI reads them via h5py instead of pynwb but accesses the same underlying HDF5 datasets.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50 ms bins spanning [-2.5, 1.5) relative to go cue. For each good unit, `np.searchsorted` is used on a `(n_trials, 81)` edge matrix to count spikes per bin. Counts are divided by bin width to get firing rates in Hz. Rates are stored as `float16`.

ii.
```python
trial_edge_matrix = np.empty((n_trials, N_BINS + 1), dtype=np.float64)
...
trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL

neural_tensor = np.empty((n_trials, n_good_units, N_BINS), dtype=np.float16)
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    counts = np.diff(edge_idx, axis=1)
    neural_tensor[:, unit_idx, :] = (counts / BIN_WIDTH_S).astype(np.float16)
```

iii. The binning approach is standard: searchsorted gives cumulative spike counts at bin edges, differencing gives per-bin counts, dividing by bin width gives Hz. The use of `float16` is a space optimization to keep the output file manageable. The AI notes this reduces the full dataset from ~11 GB to ~4.6 GB.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. No additional quality metric thresholds are applied. Sessions with zero good units are dropped entirely. This yields 69,453 good units across 173 sessions.

ii.
```python
classification = decode_strings(f["units"]["classification"])
good_unit_mask = classification == "good"
n_good_units = int(good_unit_mask.sum())
if n_good_units == 0:
    return None
```

iii. The AI correctly identifies `classification` as the classifier-based QC field from the spike sorting quality control paper, and notes that `unit_quality` would be too permissive (154,948 units vs 69,453). The count is close to the paper's 69,943.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The bin edges are computed relative to the go cue: `go_time + BIN_EDGES_REL`. Since spike times and event times are on the same absolute clock, this directly aligns spikes to the go cue without any additional transformation.

ii.
```python
trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL
...
edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
```

iii. The NWB file uses a single global clock for all timestamps, so go-cue alignment is just an offset addition.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin width is 50 ms, producing 80 bins over the [-2.5, 1.5) s window. No rebinning is applied -- spike times are binned directly into the 50 ms bins.

ii.
```python
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_WIDTH_S, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_WIDTH_S / 2.0
N_BINS = len(BIN_CENTERS_REL)
```

iii. This matches the instructions exactly. The reference code uses 40 ms bins with 3.4 ms stride; the AI intentionally deviates to match the decoder task specification.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (tone onset timestamps) and go cue times. For each trial, the last `sample_start` event before the go cue within the trial interval is taken as the effective tone onset.

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

iii. An early lick replays the sample epoch, so multiple `sample_start` events can occur per trial. Taking the last one before the go cue captures the final replay. The fallback of `go_time - 1.85` is for the edge case where no sample event is found in the trial (though this never occurred in practice).

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each bin center (relative to go cue), the time from tone onset is computed as `bin_center_abs - sample_onset`, where `bin_center_abs = go_time + BIN_CENTERS_REL`.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
```

iii. This is a straightforward time difference. The result is in seconds and increases monotonically across bins within a trial.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The bin centers used for `time_from_tone` are the same grid (`go_time + BIN_CENTERS_REL`) as used for the neural data edges (`go_time + BIN_EDGES_REL`), so they are inherently aligned -- bin k of the input corresponds to bin k of the neural data.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
```

iii. Both neural and input data use the same go-cue-relative temporal grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset`, `photostim_duration`, and `photostim_power` in the trials table, along with `start_time` for converting to absolute timestamps.

ii.
```python
trial_photostim_onset = decode_strings(trials["photostim_onset"])
trial_photostim_duration = decode_strings(trials["photostim_duration"])
trial_photostim_power = decode_strings(trials["photostim_power"])
```

iii. All three photostim fields are read; `photostim_power` is used to determine whether stimulation actually occurred (non-N/A power), in addition to onset and duration.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For trials with non-N/A photostim_power, onset, and duration, the stimulation period is computed in absolute time (`start_time + onset` to `start_time + onset + duration`). A binary time series is created where bin centers falling within this interval are set to 1.0, otherwise 0.0.

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

iii. The AI additionally checks `photostim_power` (not just onset/duration) before marking a trial as stimulated. The reference only checks `photostim_onset != 'N/A'`.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The binary photostim signal is evaluated at the same bin centers (`go_time + BIN_CENTERS_REL`) as the neural data, so alignment is automatic.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
...
photostim_row = (
    (bin_centers_abs >= stim_start_abs) & (bin_centers_abs < stim_stop_abs)
).astype(np.float32)
```

iii. Same temporal grid as neural and other inputs.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trial_instruction` ('left'/'right') and `outcome` ('hit'/'miss'/'ignore') in the trials table. For `ignore` trials, the AI additionally uses `left_lick_times` and `right_lick_times` from BehavioralEvents to attempt to find a lick-based choice.

ii.
```python
def infer_choice(instruction, outcome, trial_start, trial_stop,
                 left_lick_times, right_lick_times):
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
    ...
    return CHOICE_MAP[instruction], "ignore:instruction_fallback"
```

iii. The AI derives choice from instruction x outcome for hit/miss trials (same as reference). For ignore trials, the AI uses a multi-step fallback: first lick in trial, then instructed side. The reference instead assigns a third "no lick" category (code 2) for ignore trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0 (left) or 1 (right) -- only two categories. For ignore trials, the AI attempts to infer a lick direction from raw lick events; if none found, it uses the instructed side as a fallback. The value is constant per trial, replicated across all 80 bins.

ii.
```python
CHOICE_MAP = {"left": 0, "right": 1}
...
output_row[0, :] = choice_code
```

Output values:
```python
"output_values": [
    ["left", "right"],
    ...
]
```

iii. The AI uses only 2 choice categories (left, right), with no "no lick" option. The reference uses 3 categories (left, right, no lick). For the ~13,258 ignore trials, the AI assigns either a lick-inferred direction or the instructed side, which introduces noise into the choice label.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `outcome` column of the trials table, which contains strings 'ignore', 'miss', and 'hit'.

ii.
```python
trial_outcome = decode_strings(trials["outcome"])
...
outcome_code = OUTCOME_MAP[trial_outcome[trial_idx]]
```

iii. The trials table stores outcome explicitly with exactly the three categories needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to integers: ignore=0, miss=1, hit=2. The value is constant per trial, replicated across all 80 bins.

ii.
```python
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
...
output_row[1, :] = outcome_code
```

iii. Matches the instructions' categories exactly. Same mapping as the reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which contains 'no early' and 'early'.

ii.
```python
trial_early = decode_strings(trials["early_lick"])
...
early_code = EARLY_MAP[trial_early[trial_idx]]
```

iii. The trials table stores early lick status explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to integers: no early=0, early=1. Replicated across all 80 bins per trial.

ii.
```python
EARLY_MAP = {"no early": 0, "early": 1}
...
output_row[2, :] = early_code
```

iii. Matches the instructions' categories exactly. Same mapping as the reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose data is `(n_frames, 3)` containing tongue_x, tongue_y, and tongue_likelihood, with corresponding timestamps.

ii.
```python
tongue_group = f["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_data = tongue_group["data"][:].astype(np.float64)
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
```

iii. Same source data as the reference. Both identify column 1 as y-position and column 2 as likelihood.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI applies a multi-step preprocessing pipeline:
1. Compute frame-to-frame velocity and set a 5-sigma outlier threshold.
2. Mark frames with `likelihood < 0.1` as low-likelihood.
3. Replace low-likelihood frames with the session mean tongue y.
4. Interpolate velocity-outlier frames.
5. Compute 40th and 60th percentiles of the **full processed y array** (all frames, not binned).
6. For each trial/bin, find the nearest preceding frame to each bin center using `searchsorted`.
7. Classify: 0 if below p40, 1 if >= p40 and <= p60, 2 if > p60.

ii.
```python
def process_tongue_trace(tongue_xyzl, tongue_timestamps):
    y = tongue_xyzl[:, 1].astype(np.float64, copy=True)
    likelihood = tongue_xyzl[:, 2].astype(np.float64, copy=False)
    ...
    vel_threshold = vel_mean + 5.0 * vel_std
    outlier_mask[1:] = np.isfinite(velocity) & (velocity > vel_threshold)
    low_likelihood_mask = likelihood < LIKELIHOOD_THRESHOLD  # 0.1
    ...
    processed_y[low_likelihood_mask] = session_mean_y
    ...
    processed_y[interp_mask] = np.interp(...)
    p40, p60 = np.percentile(processed_y, [40.0, 60.0])
    ...

tongue_frame_idx = np.searchsorted(tongue_timestamps, bin_centers_abs, side="right") - 1
tongue_y = processed_tongue_y[tongue_frame_idx]
tongue_bin = np.zeros(N_BINS, dtype=np.int16)
tongue_bin[tongue_y >= tongue_info["p40"]] = 1
tongue_bin[tongue_y > tongue_info["p60"]] = 2
```

iii. The AI's approach differs significantly from the reference in several ways:
- Likelihood threshold is 0.1 (reference uses 0.5)
- Low-likelihood frames are replaced with session mean (reference sets them to NaN and excludes from bin means)
- Velocity outlier detection and interpolation is added (reference does not do this)
- Percentiles are computed on all processed frames (reference computes on 50ms bin means)
- Per-trial values use nearest-frame sampling (reference uses bin-mean averaging)
- No "not visible" class (reference uses class 3 for bins with no visible frame)

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses 3 categories (0, 1, 2) based on session-level 40th and 60th percentiles. Values below p40 are class 0, between p40 and p60 are class 1, above p60 are class 2. There is no "not visible" category.

ii.
```python
tongue_bin = np.zeros(N_BINS, dtype=np.int16)
tongue_bin[tongue_y >= tongue_info["p40"]] = 1
tongue_bin[tongue_y > tongue_info["p60"]] = 2
```

Output values:
```python
["<40th_pct", "40th_to_60th_pct", ">60th_pct"]
```

iii. The instructions specify 4 categories: 0 (< 40th pct), 1 (40th-60th), 2 (> 60th), 3 (not visible). The AI only implements 3 categories, omitting the "not visible" class entirely. Since low-likelihood frames are replaced with the session mean rather than marked as missing, there are no bins without a valid value, so the 4th class never arises.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each bin center (same grid as neural data), the nearest preceding camera frame is found via `searchsorted` and its processed y-value is used. This is a nearest-frame/sample-and-hold approach rather than averaging frames within each bin.

ii.
```python
tongue_frame_idx = np.searchsorted(tongue_timestamps, bin_centers_abs, side="right") - 1
tongue_frame_idx = np.clip(tongue_frame_idx, 0, len(processed_tongue_y) - 1)
tongue_y = processed_tongue_y[tongue_frame_idx]
```

iii. The AI uses sample-and-hold at bin centers. The reference instead averages all frames within each 50 ms bin. Both use the same go-cue-relative temporal grid for alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three main cases:
- **Sessions with no good units**: `classification` values are checked; sessions with zero good units are excluded (one session dropped, giving 173).
- **Trials outside observation intervals**: Dropped if the full analysis window isn't covered by obs_intervals. Additionally, trials with all-zero neural activity are dropped.
- **Missing sample onset**: If no `sample_start` event is found before the go cue within a trial, a fallback of `go_time - 1.85` is used (never triggered in practice).
- **Tongue tracking**: Low-likelihood frames (< 0.1) are replaced with session mean; velocity outliers are interpolated. No frames are left as missing.
- **Ignore trial choice**: Falls back to lick-event detection, then instructed side.

ii.
```python
if n_good_units == 0:
    return None

if not covered:
    n_trials_dropped_outside_obs += 1
    continue

if len(sample_candidates) == 0:
    sample_onset = go_time - 1.85

processed_y[low_likelihood_mask] = session_mean_y
processed_y[interp_mask] = np.interp(...)
```

iii. The AI handles each missing-data case with a specific strategy: exclusion where data is fundamentally absent, imputation where a reasonable estimate exists, and documented fallbacks for edge cases. The approach differs from the reference, which uses NaN propagation and explicit "not visible" categories rather than imputation.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file (via h5py) dominates runtime. The full conversion takes approximately 2-3 minutes for 173 sessions. Within each session, loading the spike times buffer and tongue tracking array are the largest I/O operations, followed by per-unit searchsorted binning. The final pickle write is significant due to the 4.6 GB output.

ii. N/A (timing is reported in conversion output logs)

iii. The AI estimated ~0.81 s/session from sample runs, projecting ~2.35 minutes for the full dataset. This is well within the 15-minute budget.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops remain:
1. The per-trial Python loop that processes each trial sequentially (finding go cue, computing inputs/outputs). This could potentially be vectorized.
2. The per-unit loop for neural binning uses vectorized `searchsorted` across all trials at once, so only the unit loop remains.
3. The tongue trace per-trial sampling is done after the main trial loop.

ii.
```python
for trial_idx in range(len(trial_start)):
    ...  # per-trial processing

for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    ...
```

iii. The per-trial loop is the main candidate for vectorization. The per-unit neural binning loop cannot be easily eliminated because each unit has a different number of spikes (ragged arrays).

## 10-c. What processing does the code repeat multiple times?

i. The AI opens each NWB file twice: once in `select_files()` to pre-scan for good units, and once in `process_session()` for actual processing. At the end of `main()`, `select_files()` is called again to estimate full-conversion time even when in sample mode.

ii.
```python
def select_files(all_files, sample_mode):
    for path in all_files:
        with h5py.File(path, "r") as f:
            classification = decode_strings(f["units"]["classification"])
            ...

# In main():
session_files, excluded_sessions = select_files(all_files, sample_mode=sample_mode)
...
est_full_s = mean_session_s * max(1, len(select_files(all_files, sample_mode=False)[0]))
```

iii. The double-open of each file (pre-scan + process) is redundant I/O, and the extra `select_files` call at the end rescans all 174 files just to count valid sessions.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes velocity-based outlier detection and interpolation for the tongue trace, which is not used by the reference processing and adds complexity without clear benefit. The AI also computes and stores `plot_payload` data for non-plot sessions (though guarded by the `make_plot` flag). The `choice_sources` counter tracks detailed provenance of each choice assignment, which is metadata-only.

ii.
```python
velocity = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
vel_threshold = vel_mean + 5.0 * vel_std
outlier_mask[1:] = np.isfinite(velocity) & (velocity > vel_threshold)
```

iii. The velocity outlier detection adds processing time for every session but produces smoothing that may not be necessary for 50 ms bin averages. The reference code does not include this step.
