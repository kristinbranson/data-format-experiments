# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files in `/app/data` using `h5py` (not `pynwb`). It walks the data directory to find all `.nwb` files, then pre-filters to "curated sessions" (those with at least one `classification == 'good'` unit) before processing. Each file is opened with `h5py.File` and the relevant groups (`intervals/trials`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`, `units`, `general/extracellular_ephys/electrodes`) are read directly.

ii.
```python
def list_session_files() -> list[str]:
    files = []
    for dirpath, _, filenames in os.walk(DATA_ROOT):
        for name in filenames:
            if name.endswith(".nwb"):
                files.append(os.path.join(dirpath, name))
    return sorted(files)

def session_has_good_units(path: str) -> bool:
    with h5py.File(path, "r") as h5:
        cls = decode_vector(h5["units"]["classification"][:])
    return "good" in cls

def discover_curated_sessions() -> list[str]:
    files = list_session_files()
    curated = [path for path in files if session_has_good_units(path)]
    return curated
```

```python
with h5py.File(path, "r") as h5:
    trials = h5["intervals"]["trials"]
    ...
    units = h5["units"]
    ...
```

iii. The AI chose `h5py` over `pynwb` to "stay close to the provided source data and avoid extra dependency / serialization layers" (CONVERSION_NOTES Step 6). The two-pass approach (first discover curated sessions, then process them) opens each file twice but achieves the same result as single-pass with a None return for sessions without good units.

## 1-b. How are the data split into subjects?

i. Subject IDs are derived from the directory name (e.g., `sub-440956`) rather than from a field inside the NWB file.

ii.
```python
def get_subject_id(path: str) -> str:
    return os.path.basename(os.path.dirname(path))
```

iii. The AI noted in CONVERSION_NOTES Step 5 that subject folder names `sub-xxxxx` are used as subject IDs. This gives subject IDs like `sub-440956` rather than numeric IDs like `440956` (from `nwb.subject.subject_id`) or mouse names like `SC015` (from `nwb.identifier`).

## 1-c. How are the data split into sessions?

i. One NWB file is one session. The session ID is derived from the filename by stripping the suffix.

ii.
```python
def get_session_id(path: str) -> str:
    return os.path.basename(path).replace("_behavior+ecephys+ogen.nwb", "").replace("_behavior+ecephys.nwb", "")
```

iii. Sessions are identified by filename rather than by the `nwb.identifier` field inside the file. The AI processes all curated sessions (those with at least one good unit), yielding 173 sessions.

## 1-d. How are the data split into trials?

i. Trials come from `h5["intervals"]["trials"]`, which is the NWB trials table. The AI reads trial metadata columns (`start_time`, `stop_time`, `outcome`, `early_lick`, etc.) and event streams (`go_start_times`, `sample_start_times`, `left_lick_times`, `right_lick_times`).

ii.
```python
trials = h5["intervals"]["trials"]
trial_start = np.asarray(trials["start_time"][:], dtype=np.float64)
trial_stop = np.asarray(trials["stop_time"][:], dtype=np.float64)
...
go_times = np.asarray(ev["go_start_times"]["timestamps"][:], dtype=np.float64)
```

iii. The AI uses the NWB trials table rows directly, with one go-cue event per trial, consistent with the data structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several trial filters:
1. Trials must be in the `obs_intervals` coverage (matched by both `start_time` AND `stop_time`)
2. `auto_water == 1` trials are excluded
3. `free_water == 1` trials are excluded
4. Trials where the decoder window `[go-2.5, go+1.5]` falls outside tongue tracking timestamps are excluded
5. Trials with all-zero neural matrices after binning are dropped
6. Sessions must have >= 2 surviving trials

ii.
```python
for i in range(len(trial_start)):
    if i not in recorded_trial_set:
        continue
    if auto_water[i] == 1 or free_water[i] == 1:
        continue
    go = float(go_times[i])
    edges_abs = go + BIN_EDGES_REL
    if edges_abs[0] < tongue_timestamps[0] or edges_abs[-1] > tongue_timestamps[-1]:
        continue
    sample_start = last_event_before(sample_start_times, float(trial_start[i]), go)
    if sample_start is None:
        continue
    ...

# Later, after neural binning:
if not np.any(neural):
    dropped_all_zero_neural += 1
    continue
```

iii. The AI documented these filters in CONVERSION_NOTES Step 5: "Exclude auto-water and free-water trials: These are atypical task contingencies that the reference analyses usually exclude." The tongue coverage filter and sample-start availability filter are additional constraints not present in the reference solution.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (the flat spike time array) and `units/spike_times_index` (the ragged index), read via h5py. Only units with `classification == 'good'` are used.

ii.
```python
spike_flat = np.asarray(units["spike_times"][:], dtype=np.float64)
spike_index = np.asarray(units["spike_times_index"][:], dtype=np.int64)
...
good_units = [i for i, c in enumerate(classification) if c == "good"]
...
unit_spike_times = [ragged_rows(spike_flat, spike_index, unit_idx) for unit_idx in good_units]
```

iii. Same source variable as the reference solution.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50 ms bins spanning [-2.5, 1.5] s relative to the go cue. Spike counts are converted to firing rates in Hz by dividing by the bin width (0.05 s). No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
def bin_spike_rates(spike_times_abs: np.ndarray, trial_edges_abs: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(spike_times_abs, trial_edges_abs, side="left")
    counts = np.diff(idx)
    return counts.astype(np.float32) / BIN_SIZE
```

```python
for unit_row, spikes_abs in enumerate(unit_spike_times):
    neural[unit_row] = bin_spike_rates(spikes_abs, edges_abs)
```

iii. The processing matches the reference approach: searchsorted-based spike counting and division by bin width to get Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are retained. Sessions with no good units are excluded during the `discover_curated_sessions` pre-filtering step.

ii.
```python
def session_has_good_units(path: str) -> bool:
    with h5py.File(path, "r") as h5:
        cls = decode_vector(h5["units"]["classification"][:])
    return "good" in cls

good_units = [i for i, c in enumerate(classification) if c == "good"]
if not good_units:
    raise ValueError(f"{session_id}: no classifier-good units")
```

iii. The AI documented this decision: "`classification == 'good'` yields 69,453 units, close to the paper's 69,943. Therefore `classification` is the correct NWB analog of the paper's good-unit list." (CONVERSION_NOTES Step 4)

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are computed as absolute times by adding go-cue-relative edges to each trial's go-cue time. Spike times and go-cue times are already on the same session-absolute clock.

ii.
```python
go = float(go_times[trial_idx])
edges_abs = go + BIN_EDGES_REL
...
neural[unit_row] = bin_spike_rates(spikes_abs, edges_abs)
```

iii. Same alignment approach as the reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins spanning [-2.5, 1.5] s relative to go cue. No rebinning is applied - spikes are directly binned into the 50 ms grid from raw spike times.

ii.
```python
WINDOW_START = -2.5
WINDOW_END = 1.5
BIN_SIZE = 0.05
NBINS = int(round((WINDOW_END - WINDOW_START) / BIN_SIZE))
BIN_EDGES_REL = np.linspace(WINDOW_START, WINDOW_END, NBINS + 1, dtype=np.float64)
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
```

iii. Matches the instructions. The AI uses `np.linspace` to generate edges while the reference uses `T_START + BIN * np.arange(N_BINS + 1)`. Both produce the same 81 edges.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (tone onset events) and go-cue times. The AI finds the last `sample_start` before each trial's go cue using `last_event_before`.

ii.
```python
sample_start_times = np.asarray(ev["sample_start_times"]["timestamps"][:], dtype=np.float64)
...
sample_start = last_event_before(sample_start_times, float(trial_start[i]), go)
```

```python
def last_event_before(times: np.ndarray, lo: float, hi: float) -> float | None:
    idx = np.searchsorted(times, hi, side="right") - 1
    if idx < 0:
        return None
    t = float(times[idx])
    if t < lo or t > hi:
        return None
    return t
```

iii. The AI constrains the search to events within [trial_start, go], while the reference solution searches all sample_start_times before go globally. Both should yield the same result since sample events within a trial occur between trial_start and go.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Time from tone onset is computed as `BIN_CENTERS_REL - tone_on_rel` where `tone_on_rel = sample_start - go`. This is equivalent to `BIN_CENTERS_REL + (go - sample_start)`, matching the reference formula `CENTERS + (go - tone)`.

ii.
```python
tone_on_rel = float(candidate_sample_onsets_rel[kept_idx])
input_arr[0] = (BIN_CENTERS_REL - tone_on_rel).astype(np.float32)
```

Where `tone_on_rel` is computed earlier as:
```python
candidate_sample_onsets_rel.append(sample_start - go)
```

iii. Algebraically equivalent to the reference: `BIN_CENTERS - (sample_start - go) = BIN_CENTERS + (go - sample_start)`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both inputs and neural data use the same go-cue-relative bin grid (`BIN_EDGES_REL` / `BIN_CENTERS_REL`), ensuring alignment.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
...
input_arr[0] = (BIN_CENTERS_REL - tone_on_rel).astype(np.float32)
```

iii. Same alignment approach as the reference.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, combined with `trial_start` to compute absolute stimulation onset/offset times.

ii.
```python
photo_onset_text = decode_vector(trials["photostim_onset"][:])
photo_dur_text = decode_vector(trials["photostim_duration"][:])
...
if photo_onset_text[i] != "N/A" and photo_dur_text[i] != "N/A":
    stim_on = float(trial_start[i]) + float(photo_onset_text[i])
    stim_off = stim_on + float(photo_dur_text[i])
```

iii. Same source variables as the reference.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time-varying input: 1 when the bin center falls within [stim_on, stim_off), 0 otherwise. Non-stimulated trials have all zeros.

ii.
```python
if math.isnan(stim_on):
    input_arr[1] = 0.0
else:
    abs_centers = go + BIN_CENTERS_REL
    input_arr[1] = ((abs_centers >= stim_on) & (abs_centers < stim_off)).astype(np.float32)
```

iii. Same logic as the reference, except the AI uses absolute times for the comparison while the reference uses go-cue-relative times. Both produce the same result.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The AI computes absolute bin centers as `go + BIN_CENTERS_REL` and compares against absolute stim_on/stim_off. This ensures the same bin grid alignment as neural data.

ii.
```python
abs_centers = go + BIN_CENTERS_REL
input_arr[1] = ((abs_centers >= stim_on) & (abs_centers < stim_off)).astype(np.float32)
```

iii. Aligned via the shared go-cue-relative bin grid, same as the reference.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from the raw lick event streams: `left_lick_times` and `right_lick_times` from `BehavioralEvents`. It finds the first lick after the go cue within a 1.5 s response window.

ii.
```python
left_lick_times = np.asarray(ev["left_lick_times"]["timestamps"][:], dtype=np.float64)
right_lick_times = np.asarray(ev["right_lick_times"]["timestamps"][:], dtype=np.float64)
...
def derive_choice_per_trial(go_times, left_lick_times, right_lick_times):
    choice = np.full(len(go_times), 2, dtype=np.int64)  # default no lick
    for i, go in enumerate(go_times):
        end = go + RESPONSE_WINDOW
        left = left_lick_times[(left_lick_times >= go) & (left_lick_times < end)]
        right = right_lick_times[(right_lick_times >= go) & (right_lick_times < end)]
        if len(left) == 0 and len(right) == 0:
            continue
        if len(left) > 0 and (len(right) == 0 or left[0] < right[0]):
            choice[i] = 0
        elif len(right) > 0 and (len(left) == 0 or right[0] < left[0]):
            choice[i] = 1
    return choice
```

iii. The AI chose to reconstruct choice from lick events because "NWB trials table has `trial_instruction` and session-level left/right lick events, but no explicit per-trial choice column" (CONVERSION_NOTES Step 4). The reference solution instead derives choice from `trial_instruction` x `outcome` (hit = instructed side, miss = opposite side, ignore = no lick).

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0 (left), 1 (right), 2 (no lick) and broadcast across all 80 bins.

ii.
```python
output_arr[0] = choice[trial_idx]
```

iii. Same coding scheme as the reference. The broadcast to all bins makes it a per-trial value repeated across time.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, which contains `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
outcome_text = decode_vector(trials["outcome"][:])
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcome_text], dtype=np.int64)
```

iii. Same source as the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: ignore=0, miss=1, hit=2, broadcast across all 80 bins.

ii.
```python
output_arr[1] = outcome[trial_idx]
```

iii. Same as the reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which contains `'no early'` and `'early'`.

ii.
```python
early_text = decode_vector(trials["early_lick"][:])
early_map = {"no early": 0, "early": 1}
early = np.array([early_map[x] for x in early_text], dtype=np.int64)
```

iii. Same source as the reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to integers: no=0, yes=1, broadcast across all 80 bins.

ii.
```python
output_arr[2] = early[trial_idx]
```

iii. Same as the reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `Camera0_side_TongueTracking` in `acquisition/BehavioralTimeSeries`. The data array is `(n_frames, 3)` containing tongue_x, tongue_y, and tongue_likelihood, with associated timestamps.

ii.
```python
tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_timestamps = np.asarray(tongue_group["timestamps"][:], dtype=np.float64)
tongue_data = np.asarray(tongue_group["data"][:], dtype=np.float64)
```

iii. Same source as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI applies several processing steps:
1. Likelihood thresholding at 0.9 (reference uses 0.5)
2. 5-sigma velocity-based outlier rejection (reference does not do this)
3. Session-wide percentiles (40th/60th) computed on raw visible frame y-values (reference computes percentiles on 50ms bin means)
4. Per-trial discretization using last-frame-in-bin sample-and-hold (reference uses mean of visible frames per bin)

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
TONGUE_VELOCITY_SIGMA = 5.0

def velocity_outlier_mask(xy, dt):
    ...
    thresh = mu + TONGUE_VELOCITY_SIGMA * sigma
    ...

def build_tongue_session_stats(timestamps, data):
    ...
    visible = np.isfinite(xy[:, 1]) & np.isfinite(likelihood) & (~outliers) & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
    visible_y = xy[visible, 1]
    q40, q60 = np.quantile(visible_y, [0.4, 0.6])
    return xy, likelihood, visible, float(q40), float(q60)

def discretize_tongue_bins(...):
    for b in range(NBINS):
        ...
        last = end - 1
        if not tongue_visible_mask[last]:
            continue
        y = float(tongue_xy[last, 1])
        if y < q40:
            out[b] = 0
        elif y <= q60:
            out[b] = 1
        else:
            out[b] = 2
    return out
```

iii. The AI justified the velocity outlier rejection by citing the method paper: "behavioral markers were tracked from video and cleaned by 5-sigma velocity-based outlier rejection" (CONVERSION_NOTES Step 3). The last-frame-in-bin approach was described as matching the reference marker alignment code: `align_markers.py` "per bin copies the last frame with timestamp in [t-dt, t)" (CONVERSION_NOTES Step 1). The 0.9 likelihood threshold was justified as: "Preliminary likelihood distributions are strongly bimodal... so a high threshold such as 0.9 is appropriate and robust" (CONVERSION_NOTES Step 5).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles of visible (likelihood >= 0.9, no velocity outlier) raw y-values define class boundaries:
- 0: y < 40th percentile
- 1: 40th <= y <= 60th percentile
- 2: y > 60th percentile
- 3: not visible (no visible frame in bin)

ii.
```python
q40, q60 = np.quantile(visible_y, [0.4, 0.6])
...
if y < q40:
    out[b] = 0
elif y <= q60:
    out[b] = 1
else:
    out[b] = 2
```

iii. The class boundaries follow the instructions. The reference computes percentiles on bin means rather than raw frames, and uses strict `<` for the 60th percentile boundary (via `np.digitize`). The AI uses `<=` for the 60th percentile boundary, which slightly differs.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Frame timestamps are on the same absolute clock as spike times and go cues. For each trial, absolute bin edges are used with `searchsorted` to find frame indices, and the last frame in each bin is used for classification.

ii.
```python
def discretize_tongue_bins(..., trial_edges_abs):
    frame_idx = np.searchsorted(frame_timestamps, trial_edges_abs)
    for b in range(NBINS):
        start = int(frame_idx[b])
        end = int(frame_idx[b + 1])
        if end <= start:
            continue
        last = end - 1
        ...
```

iii. The bin edges are the same go-cue-relative grid used for neural data, ensuring alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- Sessions with no good units: pre-filtered out during `discover_curated_sessions`
- Trials without ephys coverage: filtered via `obs_intervals` matching
- Auto-water and free-water trials: excluded
- Trials where tongue tracking doesn't cover the full decoder window: excluded
- Trials with all-zero neural after binning: dropped
- Trials without a sample_start before go: excluded
- Tongue frames with low likelihood or velocity outliers: marked not visible

ii.
```python
if edges_abs[0] < tongue_timestamps[0] or edges_abs[-1] > tongue_timestamps[-1]:
    continue
sample_start = last_event_before(sample_start_times, float(trial_start[i]), go)
if sample_start is None:
    continue
...
if not np.any(neural):
    dropped_all_zero_neural += 1
    continue
```

iii. The AI applies more aggressive filtering than the reference, which only filters by obs_intervals and free_water. The tongue coverage filter removes 765 additional trials, and the auto_water filter removes additional trials beyond free_water.

## 10-a. What are the most time-consuming steps of the code?

i. The full conversion took ~298 seconds for 173 sessions. The main cost is per-trial spike binning for every good unit (looping over units and trials), plus reading large NWB arrays from disk with h5py. The two-pass session discovery (first checking for good units, then full processing) adds overhead.

ii. N/A (runtime characteristic, not specific code)

iii. The AI noted in CONVERSION_NOTES Step 6: "The main cost is repeated per-trial spike binning for every good unit; this scales with n_trials * n_units."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops:
1. The per-trial loop over candidate trials, with nested per-unit spike binning inside: `for kept_idx, trial_idx in enumerate(candidate_trial_idx)` containing `for unit_row, spikes_abs in enumerate(unit_spike_times)`. The reference vectorizes the trial dimension by flattening all trial edges into one searchsorted call per unit.
2. The per-bin loop in `discretize_tongue_bins` iterates over each of the 80 bins.
3. The `derive_choice_per_trial` function loops over trials in Python.

ii.
```python
for kept_idx, trial_idx in enumerate(candidate_trial_idx):
    ...
    for unit_row, spikes_abs in enumerate(unit_spike_times):
        neural[unit_row] = bin_spike_rates(spikes_abs, edges_abs)
```

```python
for b in range(NBINS):
    start = int(frame_idx[b])
    end = int(frame_idx[b + 1])
    ...
```

iii. The reference solution vectorizes the trial dimension for spike binning by computing all trial edges at once: `edges = (go[:, None] + REL_EDGES[None, :]).ravel()` and doing one searchsorted per unit for all trials simultaneously. The AI's approach calls searchsorted once per unit per trial instead.

## 10-c. What processing does the code repeat multiple times?

i. The session discovery opens every NWB file once to check for good units, then opens curated files again for full processing - so each curated session file is opened twice. Within a session, no processing is repeated.

ii.
```python
def discover_curated_sessions() -> list[str]:
    files = list_session_files()
    curated = [path for path in files if session_has_good_units(path)]
    return curated
```

iii. The reference opens each file only once, returning None for sessions without good units.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes a velocity outlier mask for tongue tracking, which is not part of the reference solution's processing and adds computation. The `parse_target_region` function parses electrode JSON locations as a fallback for region labels, which is rarely used. The choice derivation from raw lick events is more complex than necessary since choice can be derived from trial_instruction + outcome.

ii.
```python
def velocity_outlier_mask(xy, dt):
    ...
    diffs = np.diff(xy, axis=0)
    speed = np.linalg.norm(diffs, axis=1) / max(dt, 1e-9)
    ...
```

iii. The velocity outlier rejection adds processing that the reference does not perform. While inspired by the method paper's marker cleaning, it applies to a different analysis context.
