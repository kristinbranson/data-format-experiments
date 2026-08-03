# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data from NWB files organized under `data/sub-*/*.nwb`. It uses `h5py` to directly read HDF5 groups rather than the `pynwb` library. Each NWB file corresponds to one session. The AI pre-filters sessions by opening each file to check for `classification == "good"` units before full processing.

ii. Finding and pre-filtering files:
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
```

Loading one session:
```python
with h5py.File(path, "r") as f:
    classification = decode_strings(f["units"]["classification"])
    ...
    trials = f["intervals"]["trials"]
    events = f["acquisition"]["BehavioralEvents"]
    go_times = events["go_start_times"]["timestamps"][:].astype(np.float64)
```

iii. The AI chose `h5py` over `pynwb` for speed and lower overhead, as noted in CONVERSION_NOTES.md Step 6. The AI also pre-scans files to exclude sessions with zero good units before full processing, effectively opening each file twice (once for filtering, once for processing).

## 1-b. How are the data split into subjects?

i. Each NWB file's subject ID is read from `general/subject/subject_id`. The AI prepends `"sub-"` to the numeric subject ID string. Subjects are collected in encounter order during assembly, not sorted.

ii. Per session:
```python
subject_id = str(f["general"]["subject"]["subject_id"][()])
if subject_id.startswith("b'"):
    subject_id = subject_id[2:-1]
subject_id = f"sub-{subject_id}"
```

At assembly:
```python
if result.subject_id not in subject_to_idx:
    subject_to_idx[result.subject_id] = len(subjects)
    subjects.append(result.subject_id)
subject_idx.append(subject_to_idx[result.subject_id])
```

iii. The AI adds a `sub-` prefix to match the directory naming convention. CONVERSION_NOTES.md Step 5 states: "Session order follows sorted NWB file list after excluding the zero-good-unit session."

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. The AI derives a session ID from the filename by stripping suffixes like `_behavior+ecephys+ogen`. Sessions are processed in sorted file order. One session is excluded for having zero good units.

ii.
```python
session_id = path.stem.replace("_behavior+ecephys+ogen", "").replace(
    "_behavior+ecephys", ""
)
```

iii. CONVERSION_NOTES.md confirms 173 sessions after excluding the one with all-NaN classification values.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`). For each trial, the go cue is found by searching `go_start_times` events that fall within the trial's `[start_time, stop_time]` interval. When multiple go cues exist in a trial interval, the last one is used.

ii.
```python
trial_start = trials["start_time"][:].astype(np.float64)
trial_stop = trials["stop_time"][:].astype(np.float64)
...
go_candidates = interval_values(go_times, start, stop)
if len(go_candidates) == 0:
    raise ValueError(f"{path.name}: no go cue found for trial {trial_idx}")
go_time = float(go_candidates[-1])
```

iii. The AI matches go cues to trials by timestamp range rather than assuming a 1:1 row correspondence. CONVERSION_NOTES.md Step 5 notes: "One go cue per trial, found by matching event timestamp into `[trial.start_time, trial.stop_time]`."

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two stages: (1) trials whose full `[-2.5, +1.5]` s go-aligned window is not covered by any observation interval of the first good unit are excluded; (2) after neural binning, trials with all-zero neural activity are dropped. No explicit `free_water` filter is applied.

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
```

iii. CONVERSION_NOTES.md Step 9 states: "The converted trial count is lower than the raw 173-session archive count because trials are excluded unless the full `[-2.5, +1.5] s` window lies inside a good-unit observation interval." The AI also drops trials with all-zero neural activity as a secondary filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (the sorted spike times of each unit). Only units with `classification == 'good'` are used. Go cue times from `BehavioralEvents/go_start_times` are used for alignment.

ii.
```python
spike_times_flat = f["units"]["spike_times"][:]
spike_times_index = f["units"]["spike_times_index"][:]
spike_times_ragged = split_ragged(spike_times_flat, spike_times_index)
good_unit_indices = np.flatnonzero(good_unit_mask)
good_spike_times = [np.asarray(spike_times_ragged[i], dtype=np.float64) for i in good_unit_indices]
```

iii. Same as the reference approach - spike times are the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50 ms bins spanning [-2.5, 1.5) s relative to the go cue. For each unit, `np.searchsorted` is applied against a trial edge matrix to get spike counts per bin, then counts are divided by bin width to get firing rates in Hz. Rates are stored as `float16`.

ii.
```python
neural_tensor = np.empty((n_trials, n_good_units, N_BINS), dtype=np.float16)
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    counts = np.diff(edge_idx, axis=1)
    neural_tensor[:, unit_idx, :] = (counts / BIN_WIDTH_S).astype(np.float16)
```

iii. CONVERSION_NOTES.md confirms the approach matches the reference code's `sliding_histogram(..., rate=True)` logic, with intentional deviation to 50 ms bins per the task instructions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept. Sessions with zero good units are excluded entirely. This yields 69,453 units across 173 sessions.

ii.
```python
classification = decode_strings(f["units"]["classification"])
good_unit_mask = classification == "good"
n_good_units = int(good_unit_mask.sum())
if n_good_units == 0:
    return None
```

iii. CONVERSION_NOTES.md Step 4 documents that `classification == "good"` matches the QC classifier described in the white paper and produces counts close to the published 69,943 (0.7% discrepancy attributed to archive version differences).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are computed as go cue time plus relative offsets. The go cue time for each trial is found by searching within the trial's time interval. Spikes are binned against these absolute edges using `searchsorted`.

ii.
```python
trial_edge_matrix[keep_idx] = rec["go_time"] + BIN_EDGES_REL
...
edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
counts = np.diff(edge_idx, axis=1)
```

iii. The AI uses the same go-cue alignment as the reference code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins spanning [-2.5, 1.5) s. No rebinning is needed since rates are computed directly from spike times.

ii.
```python
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_EDGES_REL = np.arange(WINDOW_START_S, WINDOW_END_S + 1e-9, BIN_WIDTH_S, dtype=np.float64)
```

iii. Matches the task instructions exactly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` event timestamps and the go cue time. The tone onset for a trial is the last `sample_start_times` event before the go cue, searched within `[trial_start, go_time]`.

ii.
```python
sample_candidates = interval_values(sample_start_times, start, go_time)
if len(sample_candidates) == 0:
    sample_onset = go_time - 1.85
    n_missing_sample_onset_fallback += 1
else:
    sample_onset = float(sample_candidates[-1])
```

iii. CONVERSION_NOTES.md Step 5: "For early-lick replay trials, define tone onset as the last sample-start event before go."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each bin, the value is `bin_center_absolute - sample_onset`. A fallback of `go_time - 1.85` is used if no sample onset is found (never triggered in practice).

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
```

iii. No additional processing beyond computing the time difference.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin center grid (go cue + relative bin centers), so they are inherently aligned.

ii.
```python
bin_centers_abs = go_time + BIN_CENTERS_REL
time_from_tone = (bin_centers_abs - sample_onset).astype(np.float32)
```

iii. The same time grid is used for both neural and input data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset`, `photostim_duration`, and `photostim_power` in the trials table, plus `start_time` for converting trial-relative to absolute times.

ii.
```python
trial_photostim_onset = decode_strings(trials["photostim_onset"])
trial_photostim_duration = decode_strings(trials["photostim_duration"])
trial_photostim_power = decode_strings(trials["photostim_power"])
```

iii. The AI checks all three fields (onset, duration, power) to determine if photostim occurred, while the reference only checks onset.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is 1 if its center falls between the stimulation onset and offset (onset + duration), both converted to absolute times. Trials with `N/A` power, onset, or duration have all-zero photostim.

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

iii. The AI converts photostim timing to absolute times and creates a binary time series, matching the reference approach.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Photostim onset/offset are expressed in absolute time and compared against absolute bin centers, which share the same go-cue-relative grid as the neural data.

ii.
```python
stim_start_abs = start + stim_onset
...
photostim_row = (
    (bin_centers_abs >= stim_start_abs) & (bin_centers_abs < stim_stop_abs)
).astype(np.float32)
```

iii. Same grid alignment as neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trial_instruction` and `outcome` for hit/miss trials. For ignore trials, the AI attempts to infer choice from `left_lick_times` and `right_lick_times` event streams; if no licks are found, it falls back to the instructed side.

ii.
```python
def infer_choice(instruction, outcome, trial_start, trial_stop, left_lick_times, right_lick_times):
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

iii. CONVERSION_NOTES.md Step 5 Key Decision 6: "Because the source does not contain an explicit choice label when no response occurs, use the earliest lick side in the trial if available; otherwise fall back to instructed side."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0 (left) or 1 (right) only -- there is no "no lick" class. The AI's `output_values` for choice lists only `["left", "right"]`. The value is broadcast across all 80 time bins.

ii.
```python
CHOICE_MAP = {"left": 0, "right": 1}
...
"output_values": [
    ["left", "right"],
    ...
],
```

iii. The AI always assigns a left/right choice even for ignore trials, using lick events or instructed side as fallback.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, which contains `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
trial_outcome = decode_strings(trials["outcome"])
...
outcome_code = OUTCOME_MAP[trial_outcome[trial_idx]]
```

iii. Directly from the raw trial table, same as reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: ignore=0, miss=1, hit=2. Broadcast across all 80 time bins.

ii.
```python
OUTCOME_MAP = {"ignore": 0, "miss": 1, "hit": 2}
...
output_row[1, :] = outcome_code
```

iii. Matches the instructions exactly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, containing `'no early'` and `'early'`.

ii.
```python
trial_early = decode_strings(trials["early_lick"])
...
early_code = EARLY_MAP[trial_early[trial_idx]]
```

iii. Same as reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to integers: no early=0, early=1. Broadcast across all 80 time bins.

ii.
```python
EARLY_MAP = {"no early": 0, "early": 1}
...
output_row[2, :] = early_code
```

iii. Matches the instructions exactly.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `BehavioralTimeSeries/Camera0_side_TongueTracking`, which has `(n_frames, 3)` columns for x, y, and likelihood, plus timestamps.

ii.
```python
tongue_group = f["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_data = tongue_group["data"][:].astype(np.float64)
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
```

iii. Same source variable as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI applies extensive preprocessing: (1) frames with likelihood < 0.1 have their y-value replaced with the session mean; (2) a 5-sigma velocity outlier filter is applied and outlier frames are linearly interpolated; (3) percentiles (40th, 60th) are computed on the full processed y-trace (not on bin means); (4) the processed y is sampled at each bin center (nearest preceding frame via searchsorted) and discretized into 3 classes (0: < 40th pct, 1: 40th-60th pct, 2: > 60th pct). There is no "not visible" class.

ii.
```python
def process_tongue_trace(tongue_xyzl, tongue_timestamps):
    y = tongue_xyzl[:, 1].astype(np.float64, copy=True)
    likelihood = tongue_xyzl[:, 2].astype(np.float64, copy=False)
    ...
    vel_threshold = vel_mean + 5.0 * vel_std
    ...
    processed_y[low_likelihood_mask] = session_mean_y
    ...
    processed_y[interp_mask] = np.interp(...)
    p40, p60 = np.percentile(processed_y, [40.0, 60.0])
    return processed_y.astype(np.float32), info
```

Per-trial discretization:
```python
tongue_frame_idx = np.searchsorted(tongue_timestamps, bin_centers_abs, side="right") - 1
tongue_frame_idx = np.clip(tongue_frame_idx, 0, len(processed_tongue_y) - 1)
tongue_y = processed_tongue_y[tongue_frame_idx]
tongue_bin = np.zeros(N_BINS, dtype=np.int16)
tongue_bin[tongue_y >= tongue_info["p40"]] = 1
tongue_bin[tongue_y > tongue_info["p60"]] = 2
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 8: "For each 50 ms neural bin, assign tongue y from the processed continuous tongue trace at the bin center (or closest preceding frame)." The velocity outlier processing follows the method paper's description.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories using session-wide 40th and 60th percentiles of the fully processed (imputed/interpolated) tongue y trace: 0 (< 40th), 1 (40th to 60th), 2 (> 60th). No "not visible" class exists.

ii.
```python
p40, p60 = np.percentile(processed_y, [40.0, 60.0])
...
tongue_bin = np.zeros(N_BINS, dtype=np.int16)
tongue_bin[tongue_y >= tongue_info["p40"]] = 1
tongue_bin[tongue_y > tongue_info["p60"]] = 2
```

```python
"output_values": [
    ...
    ["<40th_pct", "40th_to_60th_pct", ">60th_pct"],
],
```

iii. The AI computes percentiles on the raw processed frame-level y values rather than on 50 ms bin means. It also imputes all low-likelihood frames with the session mean before computing percentiles, which means most frames contribute to the percentile calculation even when the tongue is not visible.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each bin center (in absolute time), the nearest preceding tongue frame is found via `searchsorted`. The processed y-value at that frame is used.

ii.
```python
tongue_frame_idx = np.searchsorted(tongue_timestamps, bin_centers_abs, side="right") - 1
tongue_frame_idx = np.clip(tongue_frame_idx, 0, len(processed_tongue_y) - 1)
tongue_y = processed_tongue_y[tongue_frame_idx]
```

iii. The AI uses nearest-preceding-frame lookup rather than averaging frames within each 50 ms bin.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Session with NaN classification**: `decode_strings` converts non-string entries; no units pass `== "good"`, so session is dropped.
- **Trials outside observation intervals**: excluded by the coverage check.
- **All-zero neural trials**: dropped after neural binning (catches free_water and other edge cases).
- **Missing sample onset**: falls back to `go_time - 1.85` (never triggered in practice).
- **Ignore trials with no choice**: falls back to lick events or instructed side.
- **Low-likelihood tongue frames**: replaced with session mean y.
- **Tongue velocity outliers**: linearly interpolated.

ii.
```python
if len(sample_candidates) == 0:
    sample_onset = go_time - 1.85
    n_missing_sample_onset_fallback += 1
```

```python
nonzero_trial_mask = np.any(neural_tensor != 0, axis=(1, 2))
```

iii. CONVERSION_NOTES.md Step 10: "n_missing_sample_onset_fallback = 0" and "n_ignore_choice_fallback = 13,258."

## 10-a. What are the most time-consuming steps of the code?

i. Reading NWB files via h5py and loading spike times arrays. The full conversion completed in approximately 2-3 minutes for 173 sessions. Pickle writing was not reported as a significant bottleneck. The AI stored neural data as float16, reducing the pickle file size to 4.6 GB (vs the reference's ~11.9 GB with float32).

ii. N/A

iii. CONVERSION_NOTES.md Step 7 reports 0.81 s/session mean with projected ~2.35 min full conversion time.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops remain: (1) per-unit loop for neural binning using searchsorted across all trials at once; (2) per-trial loop for constructing trial records (go cue matching, sample onset, photostim, tongue, choice inference). The per-trial loop includes event matching and tongue sampling that could potentially be vectorized.

ii.
```python
for unit_idx, spikes in enumerate(good_spike_times):
    edge_idx = np.searchsorted(spikes, trial_edge_matrix, side="left")
    counts = np.diff(edge_idx, axis=1)
```

```python
for trial_idx in range(len(trial_start)):
    ...
    go_candidates = interval_values(go_times, start, stop)
    ...
```

iii. The per-unit loop is inherent to the ragged spike time structure. The per-trial loop handles multiple event lookups and conditional logic that makes vectorization more complex.

## 10-c. What processing does the code repeat multiple times?

i. The AI opens each NWB file twice: once in `select_files()` to check for good units, and once in `process_session()` to do the actual conversion. Also, at the end of `main()`, `select_files()` is called again to estimate full conversion time even in sample mode.

ii.
```python
def select_files(all_files: list[Path], sample_mode: bool) -> tuple[list[Path], list[str]]:
    for path in all_files:
        with h5py.File(path, "r") as f:
            classification = decode_strings(f["units"]["classification"])
            ...
```

```python
est_full_s = mean_session_s * max(1, len(select_files(all_files, sample_mode=False)[0]))
```

iii. The pre-filtering scan reads classification from all files before processing begins. The final `select_files` call at the end re-opens all files unnecessarily for time estimation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The tongue velocity outlier detection and linear interpolation is substantial processing that the reference solution does not perform. Additionally, the AI computes and stores `stim_power` but only uses it as a gate for photostim presence. The `plot_payload` dict is constructed for plotting mode but unused otherwise. The `SessionResult` dataclass stores statistics and metadata that are not all used in the final output.

ii.
```python
velocity = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
valid_velocity = np.isfinite(velocity)
if np.any(valid_velocity):
    vel_mean = np.nanmean(velocity[valid_velocity])
    vel_std = np.nanstd(velocity[valid_velocity])
    vel_threshold = vel_mean + 5.0 * vel_std
```

iii. The velocity outlier processing follows the method paper's description but is extra complexity compared to the simpler approach of just masking low-likelihood frames.
