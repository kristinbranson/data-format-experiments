# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs all NWB files under `data/sub-*/` and processes each file as one session. It uses `h5py` for direct HDF5 access rather than `pynwb`. A pre-discovery pass (`discover_usable_files`) first opens every file to check for classifier-good units, then usable files are processed sequentially in `convert_session()`.

ii.
```python
files = sorted(glob.glob(str(DATA_DIR / "sub-*" / "*.nwb")))
# ...
for path in files:
    with h5py.File(path, "r") as nwb:
        classification = nwb["units/classification"][()]
        n_good = int(np.count_nonzero(classification == b"good"))
```
```python
def convert_session(path: str) -> SessionResult:
    with h5py.File(path, "r") as nwb:
        # ...
```

iii. The AI chose `h5py` over `pynwb` for performance (avoiding materialization of irrelevant waveform/electrode data). The discovery pass pre-filters sessions without good units before the heavier conversion pass.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the file path (e.g., `sub-440956` becomes `440956`) rather than from NWB subject metadata. Unique subjects are collected from all file paths and sorted.

ii.
```python
def subject_from_path(path: str) -> str:
    return Path(path).name.split("_")[0].removeprefix("sub-")
```
```python
subjects = sorted({subject_from_path(path) for path in files})
subject_lookup = {name: i for i, name in enumerate(subjects)}
```

iii. The AI derives subject IDs from directory/file naming conventions rather than `nwb.subject.subject_id`. Both produce the same numeric identifiers since the path encodes the subject ID.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. The session ID is derived from the filename by stripping the `_behavior...` suffix.

ii.
```python
def session_id_from_path(path: str) -> str:
    return Path(path).name.split("_behavior")[0]
```

iii. The dandiset stores one session per file, so no grouping is needed. Session order follows the sorted file list.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. The AI maps ephys-backed trial rows to the trial table using `obs_intervals` start times matched via `searchsorted`.

ii.
```python
trials = nwb["intervals/trials"]
n_trial_table = int(trials["id"].shape[0])
# ...
obs_intervals = np.asarray(nwb["units/obs_intervals"][obs_lo:obs_hi], dtype=np.float64)
ephys_trial_idx = np.searchsorted(trial_starts_table, obs_intervals[:, 0])
```

iii. The AI identified that in some sessions the ephys recording doesn't cover all behavioral trials, so it maps neural trials to exact trial-table rows through observation intervals rather than assuming a prefix alignment.

## 1-e. How are trials filtered based on quality controls?

i. Multiple filters are applied sequentially: (1) observation-interval mapping restricts to ephys-backed trials; (2) `is_good_trials` per-unit validity mask requires all retained units to be valid (`valid_prefix = np.all(good_trial_matrix, axis=0)`); (3) both `auto_water` and `free_water` trials are excluded; (4) trials with all-zero neural activity across all units are removed post-hoc. A session needs at least 2 surviving trials.

ii.
```python
good_trial_matrix = np.asarray(nwb["units/is_good_trials"][unit_indices, :], dtype=bool)
valid_prefix = np.all(good_trial_matrix, axis=0)
candidate_trial_idx = ephys_trial_idx[np.flatnonzero(valid_prefix)]
auto_water = np.asarray(trials["auto_water"])[candidate_trial_idx] != 0
free_water = np.asarray(trials["free_water"])[candidate_trial_idx] != 0
water_trial = auto_water | free_water
selected_trial_idx = candidate_trial_idx[~water_trial]
# ...
zero_spike_trials = ~np.any(neural_rates > 0, axis=(1, 2))
if n_zero_spike_trials:
    keep = ~zero_spike_trials
```

iii. The AI justified the `is_good_trials` filter as honoring NWB's per-unit trial validity annotations (509 trials dropped). The `auto_water` exclusion follows the reference code's `get_regular_trial_mask` because "outcome semantics are experimentally altered." The all-zero removal catches edge cases where spike recording ended before the trial window. This yields 89,068 trials total.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (ragged spike time array) and `units/spike_times_index`, using only units where `units/classification == b"good"`. Go-cue times from `BehavioralEvents/go_start_times` define the bin placement.

ii.
```python
spikes = np.asarray(spike_times_ds[lo:hi], dtype=np.float64)
# ...
unit_indices = np.flatnonzero(classification == b"good").astype(np.int64)
```

iii. `spike_times` is the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. For each good unit, spikes are assigned to trials via `searchsorted` on trial window starts, then to 50ms bins via floor division. `np.bincount` accumulates counts per (trial, bin), then counts are divided by 0.05s to get Hz firing rates. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
def bin_selected_units(spike_times_ds, spike_index, unit_indices, go_times):
    window_starts = go_times + OFF_START
    window_ends = go_times + OFF_END
    rates = np.zeros((n_trials, unit_indices.size, N_TIME), dtype=np.float32)
    for out_unit, raw_unit in enumerate(unit_indices):
        # ...
        trial_idx = np.searchsorted(window_starts, spikes, side="right") - 1
        # ...
        bin_idx = np.floor((rel - OFF_START) / BIN_SIZE_S).astype(np.int64)
        flat_idx = trial_idx[good_bin] * N_TIME + bin_idx[good_bin]
        counts = np.bincount(flat_idx, minlength=n_trials * N_TIME)
        rates[:, out_unit, :] = counts.reshape(n_trials, N_TIME) / BIN_SIZE_S
```

iii. The approach matches the reference code's `sliding_histogram` convention: half-open bin intervals, count spikes, divide by bin width for Hz. The per-unit loop is inherent to ragged spike storage.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == b"good"` are kept, matching the QC classifier from the spike sorting white paper. Additionally, units must have a non-empty Allen CCF annotation (`anno_name`). One session with all-NaN classifier labels is excluded.

ii.
```python
classification = nwb["units/classification"][()]
unit_indices = np.flatnonzero(classification == b"good").astype(np.int64)
region_names = [decode_text(x).strip() for x in nwb["units/anno_name"][unit_indices]]
if any((not x) or x.lower() == "nan" for x in region_names):
    raise ValueError(f"{session_id}: curated unit lacks a CCF annotation")
```

iii. The AI uses the same classifier-good criterion as the reference. The CCF annotation check is an additional guard but in practice all classifier-good units have valid annotations. This retains 69,453 units across 173 sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All timestamps in the NWB file share a global session clock. Trial windows are defined as `[go_time + OFF_START, go_time + OFF_END)` for each go-cue time. Spikes are assigned to trials by their position relative to window start times.

ii.
```python
go_all = np.asarray(nwb["acquisition/BehavioralEvents/go_start_times/timestamps"])
go_times = go_all[selected_trial_idx]
# ...
window_starts = go_times + OFF_START
rel = spikes - go_times[trial_idx]
```

iii. No resampling or clock correction is needed since everything is on the same session-absolute clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 80 non-overlapping 50ms bins span -2.5s to +1.5s relative to the go cue. Bin edges are defined as `np.linspace(-2.5, 1.5, 81)`.

ii.
```python
OFF_START, OFF_END, BIN_SIZE_S = -2.5, 1.5, 0.050
BIN_EDGES = np.linspace(OFF_START, OFF_END, 81, dtype=np.float64)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
N_TIME = BIN_CENTERS.size
```

iii. The 50ms bin width and [-2.5, 1.5] window are set by the instructions. 80 timepoints per trial is consistent across all sessions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` timestamps and `trial start_time`. The AI takes the **first** `sample_start_times` event at or after each trial's `start_time`.

ii.
```python
sample_starts = np.asarray(nwb["acquisition/BehavioralEvents/sample_start_times/timestamps"])
first_sample_idx = np.searchsorted(sample_starts, trial_starts, side="left")
tone_onsets = sample_starts[first_sample_idx]
```

iii. CONVERSION_NOTES: "Define tone onset as the first sample-start event in the behavioral trial, matching 'time from tone onset' rather than the final replay. This intentionally makes elapsed time informative on replay/early-lick trials."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each bin center (in global time), subtract the trial's tone onset time. The result is the elapsed time from tone onset to each bin center, in seconds.

ii.
```python
centers_global = go_times[:, None] + BIN_CENTERS[None, :]
elapsed_from_tone = (centers_global - tone_onsets[:, None]).astype(np.float32)
```

iii. Straightforward subtraction. The formula is the same as the reference; only the tone onset selection differs (from 3-a).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both neural bins and tone-elapsed time use the same bin centers defined relative to the go cue. `centers_global = go_times[:, None] + BIN_CENTERS[None, :]`.

ii.
```python
centers_global = go_times[:, None] + BIN_CENTERS[None, :]
elapsed_from_tone = (centers_global - tone_onsets[:, None]).astype(np.float32)
```

iii. The same temporal grid is used for all data streams, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From the global event timestamps `BehavioralEvents/photostim_start_times/timestamps` and `BehavioralEvents/photostim_stop_times/timestamps`, rather than from the trials table's `photostim_onset` and `photostim_duration` columns.

ii.
```python
stim_starts = np.asarray(nwb["acquisition/BehavioralEvents/photostim_start_times/timestamps"])
stim_stops = np.asarray(nwb["acquisition/BehavioralEvents/photostim_stop_times/timestamps"])
```

iii. The AI uses global start/stop event pairs rather than per-trial onset+duration from the trials table. Both should produce equivalent stimulation intervals on the global clock.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is 1 if its global center falls within a `[start, stop)` interval and 0 otherwise. The AI uses `searchsorted` on the start times to find which interval each bin center belongs to, then checks whether the center is within that interval.

ii.
```python
def photostim_state(centers_global, starts, stops):
    flat = centers_global.ravel()
    interval_idx = np.searchsorted(starts, flat, side="right") - 1
    safe_idx = np.clip(interval_idx, 0, starts.size - 1)
    on = (interval_idx >= 0) & (flat >= starts[safe_idx]) & (flat < stops[safe_idx])
    return on.reshape(centers_global.shape).astype(np.float32)
```

iii. The result is a binary time series. Non-stimulated trials have no matching intervals, so all bins are 0.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The same global bin centers (`centers_global = go_times[:, None] + BIN_CENTERS`) are used for both photostimulation state evaluation and neural binning.

ii.
```python
stim_on = photostim_state(centers_global, stim_starts, stim_stops)
```

iii. Both streams share the same temporal grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trial_instruction` (left/right) and `outcome` (hit/miss/ignore) in the trials table, since actual lick direction is not stored directly.

ii.
```python
outcomes = np.asarray(trials["outcome"])[selected_trial_idx]
instructions = np.asarray(trials["trial_instruction"])[selected_trial_idx]
```

iii. The mapping is: hit -> instruction side, miss -> opposite side, ignore -> no lick. Same logic as the reference.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Encoded as left=0, right=1, no lick=2. Per-trial values are repeated across all 80 time bins.

ii.
```python
def map_choice(instructions, outcomes):
    result = np.empty(outcomes.size, dtype=np.int8)
    for i, (instruction_raw, outcome_raw) in enumerate(zip(instructions, outcomes)):
        instruction, outcome = decode_text(instruction_raw), decode_text(outcome_raw)
        if outcome == "ignore":
            result[i] = 2
        elif outcome == "hit":
            result[i] = 0 if instruction == "left" else 1
        elif outcome == "miss":
            result[i] = 1 if instruction == "left" else 0
    return result
# ...
outputs[:, 0, :] = map_choice(instructions, outcomes)[:, None]
```

iii. The encoding matches the reference exactly: left=0, right=1, no lick=2. Output values are `["left", "right", "no lick"]`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which contains `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcomes = np.asarray(trials["outcome"])[selected_trial_idx]
```

iii. The trials table stores outcome explicitly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to ignore=0, miss=1, hit=2, repeated across all 80 bins.

ii.
```python
def map_outcome(outcomes):
    code = {"ignore": 0, "miss": 1, "hit": 2}
    return np.asarray([code[decode_text(x)] for x in outcomes], dtype=np.int8)
# ...
outputs[:, 1, :] = map_outcome(outcomes)[:, None]
```

iii. Same encoding as the reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds `'no early'` and `'early'`.

ii.
```python
early = np.asarray(trials["early_lick"])[selected_trial_idx]
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to no=0, yes=1, repeated across all 80 bins.

ii.
```python
def map_early(early):
    code = {"no early": 0, "early": 1}
    return np.asarray([code[decode_text(x)] for x in early], dtype=np.int8)
# ...
outputs[:, 2, :] = map_early(early)[:, None]
```

iii. Same encoding as the reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains `(n_frames, 3)` data with columns tongue_x, tongue_y, tongue_likelihood, plus timestamps.

ii.
```python
tracking = nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tracking_ts = np.asarray(tracking["timestamps"], dtype=np.float64)
tracking_data = np.asarray(tracking["data"], dtype=np.float64)
```

iii. Same source as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Multi-step processing: (1) Frames with likelihood < 0.9 are marked invisible. (2) A 5-sigma velocity outlier cleaning step identifies and interpolates high-confidence frames with extreme velocity. (3) Session-wide 40th and 60th percentiles are computed over **raw visible (cleaned) frames** (not bin means). (4) Per bin, the last raw frame at or before each bin center (within ~5.1ms tolerance) is sampled ("sample-and-hold"). (5) Visible frames are categorized: y < q40 -> 0, q40 <= y <= q60 -> 1, y > q60 -> 2. Bins with no nearby visible frame -> 3 (not visible).

ii.
```python
DLC_VISIBLE_THRESHOLD = 0.9
# In clean_tongue_tracking:
visible = finite & (likelihood >= DLC_VISIBLE_THRESHOLD)
# velocity cleaning:
speed[1:][valid_velocity] = np.linalg.norm(dxy[valid_velocity], axis=1) / dt[valid_velocity]
velocity_threshold = float(velocity_values.mean() + 5.0 * velocity_values.std())
# interpolate outliers:
for dim in range(2):
    xy[outlier, dim] = np.interp(timestamps[outlier], timestamps[base_valid], xy[base_valid, dim])
# percentiles from raw visible frames:
q40, q60 = np.percentile(clean_y[visible], [40.0, 60.0])
```
```python
# In sample_tongue_categories - sample-and-hold:
idx = np.searchsorted(timestamps, flat, side="right") - 1
categories[is_visible & (y < q40)] = 0
categories[is_visible & (y >= q40) & (y <= q60)] = 1
categories[is_visible & (y > q60)] = 2
```

iii. The AI applied the method paper's 5-sigma velocity cleaning for visible frames, used a 0.9 likelihood threshold (justified by the bimodal distribution), and chose sample-and-hold over bin averaging as described in CONVERSION_NOTES: "Follow the reference marker alignment's sample-and-hold convention (last raw frame at/before a requested time)."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The 40th and 60th percentiles of cleaned visible y-values (computed over raw frames, not bin means) define category boundaries: below q40 -> 0, q40 to q60 -> 1, above q60 -> 2, not visible -> 3.

ii.
```python
q40, q60 = np.percentile(clean_y[visible], [40.0, 60.0])
# ...
categories[is_visible & (y < q40)] = 0
categories[is_visible & (y >= q40) & (y <= q60)] = 1
categories[is_visible & (y > q60)] = 2
```

iii. The percentile split and class codes match the instructions. The AI computes percentiles over raw visible frames rather than over 50ms bin means (as the reference does).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each bin center (global time), the last video frame at or before that time is found via `searchsorted`. A frame is used only if it's within ~5.1ms (roughly one frame at 300Hz). This "sample-and-hold" approach differs from the reference's bin-averaging approach.

ii.
```python
def sample_tongue_categories(centers_global, timestamps, clean_y, visible, q40, q60):
    flat = centers_global.ravel()
    idx = np.searchsorted(timestamps, flat, side="right") - 1
    safe_idx = np.clip(idx, 0, timestamps.size - 1)
    age = flat - timestamps[safe_idx]
    near = (idx >= 0) & (age >= -1e-9) & (age <= 0.0051)
    is_visible = near & visible[safe_idx]
```

iii. CONVERSION_NOTES: "Follow the reference marker alignment's sample-and-hold convention (last raw frame at/before a requested time), with a one-frame tolerance. Do not average visible and invisible samples inside a 50-ms bin because that would blur the required not-visible class."

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Session with NaN/missing classifier labels: excluded in discovery pass (1 session dropped). (2) Trials outside ephys observation intervals: excluded via obs_intervals mapping. (3) Trials invalid for any retained unit: excluded via `is_good_trials`. (4) Auto/free water trials: excluded. (5) Trials with all-zero neural activity: excluded post-hoc (2 trials). (6) Invisible tongue frames: assigned class 3 ("not visible"). (7) High-confidence velocity outliers in tongue tracking: interpolated from neighboring valid frames.

ii.
```python
# Missing classifier labels
n_good = int(np.count_nonzero(classification == b"good"))
if n_good: usable.append(path)
# All-zero neural trials
zero_spike_trials = ~np.any(neural_rates > 0, axis=(1, 2))
if n_zero_spike_trials:
    keep = ~zero_spike_trials
# Velocity outlier interpolation
for dim in range(2):
    xy[outlier, dim] = np.interp(timestamps[outlier], timestamps[base_valid], xy[base_valid, dim])
```

iii. The AI is thorough in handling edge cases, with explicit guards for each type of missing/invalid data documented in CONVERSION_NOTES.

## 10-a. What are the most time-consuming steps of the code?

i. Reading NWB files via h5py dominates, followed by per-unit spike binning. The full conversion completes in ~149s for 173 sessions. The AI also performs two additional pre-passes: one to discover usable files and one to gather brain region names.

ii.
```python
# Discovery pass
for path in files:
    with h5py.File(path, "r") as nwb:
        classification = nwb["units/classification"][()]
# Region gathering pass
for path in files:
    with h5py.File(path, "r") as nwb:
        good = nwb["units/classification"][()] == b"good"
        region_set.update(...)
# Main conversion pass
for number, path in enumerate(files, 1):
    result = convert_session(path)
```

iii. The AI identified file I/O and spike binning as the main costs. Total runtime is well within the 15-minute budget.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit loop in `bin_selected_units` iterates over each good unit individually. Within the loop, spike-to-trial and spike-to-bin assignment are vectorized. The tongue sampling (`sample_tongue_categories`) is fully vectorized with no per-trial loop.

ii.
```python
for out_unit, raw_unit in enumerate(unit_indices):
    # vectorized within: searchsorted, bincount
    spikes = np.asarray(spike_times_ds[lo:hi], dtype=np.float64)
    trial_idx = np.searchsorted(window_starts, spikes, side="right") - 1
    # ...
```

iii. The per-unit loop cannot be eliminated due to ragged spike storage. The AI achieved more vectorization than the reference on tongue processing (no per-trial loop).

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened **three times**: (1) in `discover_usable_files()` to check for good units, (2) in the region-gathering loop in `main()` to collect brain region names, and (3) in `convert_session()` for the actual conversion. Classification labels are read in both the discovery pass and the conversion pass.

ii.
```python
# Pass 1: discover_usable_files
for path in files:
    with h5py.File(path, "r") as nwb:
        classification = nwb["units/classification"][()]

# Pass 2: region gathering in main()
for path in files:
    with h5py.File(path, "r") as nwb:
        good = nwb["units/classification"][()] == b"good"
        region_set.update(decode_text(x).strip() for x in nwb["units/anno_name"][good])

# Pass 3: convert_session
with h5py.File(path, "r") as nwb:
    classification = nwb["units/classification"][()]
```

iii. The multi-pass design was chosen for code clarity (separate discovery, pre-computation, and conversion stages) at the cost of redundant I/O.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `plot_payload` dictionary is computed for **every** session even when `--show-processing` is not enabled. This includes storing raw tracking data, clean tongue values, and neural rate arrays in a separate dict that is only used for plotting. Additionally, the velocity cleaning statistics (threshold, outlier count) are computed and stored in metadata but are not used by the decoder.

ii.
```python
# Always computed, even without --show-processing:
plot_payload = {
    "session_id": session_id,
    "tracking_ts": tracking_ts,
    "tracking_y_raw": tracking_data[:, 1],
    "tracking_likelihood": tracking_data[:, 2],
    "tracking_y_clean": clean_y,
    # ...
}
# Only used conditionally:
if args.show_processing and number <= 2:
    plot_processing(result.plot_payload, plot_path)
```

iii. The plot payload adds temporary memory overhead per session but doesn't affect correctness. The velocity cleaning is part of the tongue processing pipeline regardless.
