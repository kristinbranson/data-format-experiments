# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads NWB files directly using `h5py` rather than the Allen SDK. It reads the metadata CSV (`ophys_experiment_table.csv`) to discover all experiments, filters to non-passive sessions, and then iterates over each experiment's NWB file loading trial tables, neural data, running speed, pupil tracking, and stimulus presentations.

ii.
```python
def read_metadata_sessions() -> List[SessionInfo]:
    exp_table = pd.read_csv(METADATA_DIR / "ophys_experiment_table.csv")
    file_map = {
        int(path.stem.split("_")[-1]): path
        for path in sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
    }
    exp_table = exp_table[exp_table["ophys_experiment_id"].isin(file_map)].copy()
    exp_table = exp_table[~exp_table["passive"]].copy()
    ...

def read_session_raw(session: SessionInfo, load_events: bool = True) -> Dict[str, object]:
    with h5py.File(session.path, "r") as h5f:
        ...
```

iii. The AI documented that the SDK's `pynwb/hdmf` stack could not instantiate the NWB 2.6.0 files due to an `external_resources` abstract-method mismatch, so direct HDF5 reads were used instead. The AI verified field semantics matched the SDK objects.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mouse_id` from the experiment table CSV. Each unique `mouse_id` becomes a subject entry.

ii.
```python
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
```

iii. The AI uses the same `mouse_id` field as the reference. Subject assignment is done on-the-fly as sessions are processed.

## 1-c. How are the data split into sessions?

i. Each NWB file (one per `ophys_experiment_id`, i.e., one imaging plane) is treated as an independent session. This differs from the reference, which groups multiple experiments sharing the same `ophys_session_id` into a single session with merged neurons.

ii.
```python
sessions = [
    SessionInfo(
        ophys_experiment_id=int(row.ophys_experiment_id),
        path=file_map[int(row.ophys_experiment_id)],
        ...
    )
    for row in exp_table.itertuples(index=False)
]
```

iii. The AI noted in Step 4 of CONVERSION_NOTES.md: "Treat each NWB experiment file as one decoder session because neural traces are experiment-specific." This is a different choice from the reference, which merges experiments from the same ophys_session_id.

## 1-d. How are the data split into trials?

i. Trials are defined using the NWB `intervals/trials` table. For each valid trial (go or catch, not aborted, not auto-rewarded), the full trial window from `start_time` to `stop_time` is used. The AI creates a common 30 Hz time grid from `start_time` to `stop_time` for each trial.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. The AI uses the SDK's trial table directly, segmenting trials from `start_time` to `stop_time`. This is similar to the reference approach, though the reference also requires `change_time.notna()`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to keep only `(go or catch) and not aborted and not auto_rewarded`. Sessions with fewer than 2 valid trials are excluded. Additionally, sessions missing eye-tracking pupil data are excluded entirely.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
if int(raw["keep_mask"].sum()) < 2:
    ...continue
...
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
```

iii. The AI documented that passive sessions are excluded because they lack meaningful trial outcomes. Sessions missing eye tracking (3 sessions) are excluded to avoid NaN pupil data. The `go | catch` filter is functionally equivalent to the reference's `~aborted & ~auto_rewarded & change_time.notna()`, since non-aborted, non-auto-rewarded trials are either go or catch.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `processing/ophys/event_detection/data` — the precomputed calcium **events**, not `dff_traces`.

ii.
```python
events = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32
)
```

iii. The AI explicitly chose events over dF/F based on the strategy paper: "The strategy paper states its neural analyses use detected calcium events rather than raw dF/F." This is documented in Step 4 and Step 5 of CONVERSION_NOTES.md.

## 2-b. How is the `neural` data processed?

i. The event traces are linearly interpolated from the native ophys timestamps onto a common 30 Hz time grid for each trial. The interpolation is done per-trial using `interpolate_matrix()`.

ii.
```python
def interpolate_matrix(source_t, source_values, query_t):
    right = np.searchsorted(source_t, query_t, side="left")
    right = np.clip(right, 0, n_src - 1)
    left = np.clip(right - 1, 0, n_src - 1)
    ...
    out = left_vals * (1.0 - w[:, None]) + right_vals * w[:, None]
    return out.astype(np.float32)
...
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The AI justified the 30 Hz resampling based on the strategy paper: "The strategy paper linearly interpolates calcium event responses onto common 30 Hz timestamps." The reference solution does not resample; it keeps the native ophys frame rate.

## 2-c. How is the `neural` data filtered based on quality controls?

i. If `valid_roi` is present in the cell specimen table and differs from the event-trace width, traces are filtered to only valid ROIs.

ii.
```python
if load_events:
    cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cell_table = len(cell_table["id"])
    if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
        valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
        if valid_roi.sum() != events.shape[1]:
            events = events[:, valid_roi]
```

iii. The AI documented: "Keep only valid ROIs / cells from the released NWB content; do not add extra ad hoc neuron filtering beyond reference QC/filtering already reflected in the files." The reference also includes all neurons from the SDK without additional filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the trial start time. A common 30 Hz grid is created from `start_time` to `stop_time`, and event traces are linearly interpolated onto this grid.

ii.
```python
grid = session_grid(start, stop)  # 30 Hz grid from trial start to stop
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. The instructions say "Temporally align based on ophys timestamp." Both solutions align to the trial's ophys timestamps, but the AI adds a resampling step to 30 Hz while the reference keeps the native rate.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI rebins all data to a common 30 Hz grid (~33.33 ms bins). This is applied uniformly across all sessions regardless of native ophys rate.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
...
def session_grid(start, stop, dt=TIME_BIN_SIZE_S):
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
```

iii. The AI justified this based on the strategy paper's 30 Hz interpolation and the fact that behavior/eye tracking are natively at 30 Hz, making 30 Hz "the most defensible common grid." The reference keeps the native ~11 Hz ophys rate without rebinning.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the stimulus presentations table (`image_name`, `start_time`, `stop_time`, `omitted`), not from the trials table. A `gray` category is used for inter-stimulus intervals and omitted stimuli.

ii.
```python
def stimulus_identity_codes(stimulus, query_t, image_to_code):
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    image_names = stimulus["image_name"]
    omitted = stimulus["omitted"]
    idx = np.searchsorted(starts, query_t, side="right") - 1
    codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
    ...
```

iii. The AI documented: "Piecewise-constant categorical signal on 30 Hz grid; use actual image name during image display; use `gray` during gray-screen or omission periods." The reference instead derives image identity from `initial_image_name` and `change_image_name` in the trials table, without a gray category.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each time bin, the AI checks whether that time falls within a stimulus presentation window. If so, and the stimulus is not omitted, the image name is used. Otherwise, the `gray` code is assigned. Image names are mapped to integer codes via a global sorted vocabulary that includes `gray`.

ii.
```python
image_values = ["gray"] + sorted(image_names)
image_to_code = {name: idx for idx, name in enumerate(image_values)}
...
codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
# within presentation windows, assign the image code
```

iii. The AI's approach produces a time-varying image identity that alternates between specific image codes and `gray`. The reference uses only the trial-level `initial_image_name`/`change_image_name` without gray periods.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity codes are computed on the same 30 Hz trial grid as the neural data, ensuring frame-level alignment.

ii.
```python
grid = session_grid(start, stop)
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
```

iii. Both the AI and reference compute image identity on the same time grid as neural data. The AI uses the 30 Hz resampled grid; the reference uses native ophys frame indices.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus presentations table: `is_change`, `start_time`, `stop_time`, and `omitted`.

ii.
```python
def stimulus_change_codes(stimulus, query_t):
    starts = stimulus["start_time"]
    stops = stimulus["stop_time"]
    is_change = stimulus["is_change"]
    omitted = stimulus["omitted"]
    ...
```

iii. The AI uses the stimulus presentation's `is_change` flag to identify change presentations. The reference instead uses the trial's `go` flag and `change_time` to create a 750ms binary window.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each time bin within a stimulus presentation where `is_change` is True and the stimulus is not omitted, the image change code is set to 1. Otherwise it remains 0. This means image_change is 1 for the duration of the change stimulus presentation (~250ms image display).

ii.
```python
codes = np.zeros(query_t.shape, dtype=np.int64)
...
changed = is_change[sub_idx] & (~omitted[sub_idx])
codes[assign] = changed.astype(np.int64)
```

iii. The AI chose to mark the entire change-image presentation window rather than a fixed 750ms window. The reference marks a 750ms window (one flash + grey period) starting at change_time for go trials only.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Image change is binary: 0 (no change) or 1 (change). No thresholding is needed — it is directly computed as a binary variable.

ii.
```python
image_change = stimulus_change_codes(raw["stimulus"], grid)
# values are 0 or 1
```

iii. Both the AI and reference produce a binary variable. The instructions say "Have value of 1 right after a change in image identity, otherwise 0."

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same 30 Hz trial grid as neural data.

ii.
```python
grid = session_grid(start, stop)
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. Same alignment approach as image identity and neural data.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `processing/running/speed/data` and its timestamps in the NWB file.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. This is equivalent to the reference's `dataset.running_speed`, which accesses the same underlying data.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated (`np.interp`) from its native timestamps to the 30 Hz trial grid, then discretized into 5 quintile bins using globally computed bin edges.

ii.
```python
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
running_bins = digitize_with_edges(running_cont, running_edges)
...
def robust_quintile_edges(values):
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
```

iii. The AI uses quintile edges at [20, 40, 60, 80] percentiles via `np.nanpercentile`, producing 5 bins. The reference uses `np.linspace(0, 100, 6)` = [0, 20, 40, 60, 80, 100] percentiles. Both produce 5 equal-percentile bins but the edge computation differs slightly.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Running speed is discretized into 5 bins (0-4) using `np.digitize` with 4 edges at the 20th, 40th, 60th, and 80th percentiles.

ii.
```python
def robust_quintile_edges(values):
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
    for i in range(1, len(percentiles)):
        if percentiles[i] <= percentiles[i - 1]:
            percentiles[i] = percentiles[i - 1] + 1e-6
    return percentiles

def digitize_with_edges(values, edges):
    return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. The AI adds a robustness fix to ensure strictly increasing edges. The reference computes percentile-based bin edges similarly but uses `np.digitize(values, bin_edges[1:-1])`.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same 30 Hz trial grid as the neural data, ensuring alignment.

ii.
```python
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
```

iii. Both solutions interpolate running speed to the same time grid as neural data.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from `acquisition/EyeTracking/pupil_tracking/width` and `height`, using `max(width, height)` as the diameter. Timestamps come from `eye_tracking/timestamps`.

ii.
```python
pupil_width = np.asarray(
    h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], dtype=np.float32
)
pupil_height = np.asarray(
    h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["height"], dtype=np.float32
)
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```

iii. The AI uses `max(width, height)` while the reference uses only `pupil_width`. The AI documented: "Compute pupil diameter as `max(width, height)`."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Pupil diameter is interpolated from its native timestamps to the 30 Hz trial grid. Non-finite values (from blinks/artifacts) are excluded before interpolation. Then the continuous signal is discretized into 5 quintile bins.

ii.
```python
def interpolate_pupil(pupil_t, pupil_diameter, query_t):
    valid = np.isfinite(pupil_diameter)
    ...
    return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
```

iii. The AI uses `np.isfinite()` to filter invalid samples before interpolation, rather than explicitly checking a `likely_blink` flag. The reference explicitly filters by `~eye['likely_blink']` before interpolation.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same approach as running speed: 5 quintile bins (0-4) using globally computed edges at [20, 40, 60, 80] percentiles.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. Same quintile approach as running speed. Both reference and AI use 5 equal-percentile bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same 30 Hz trial grid as neural data.

ii.
```python
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. Same alignment approach as all other signals.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean columns `hit`, `miss`, `false_alarm`, and `correct_reject` in the trials table.

ii.
```python
def trial_outcome_code(trials, idx, mapping):
    for name in ("hit", "miss", "false_alarm", "correct_reject"):
        if bool(trials[name][idx]):
            return mapping[name]
    raise ValueError(f"Trial {idx} has no valid outcome label")
```

iii. This matches the reference approach exactly.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Trial outcomes are mapped to integer codes (0-3) via a fixed mapping: hit=0, miss=1, false_alarm=2, correct_reject=3. The code is broadcast across all time bins within the trial.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
...
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. Both the AI and reference map outcomes identically and broadcast across time.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Missing eye tracking**: 3 active sessions without pupil data are excluded entirely.
- **Invalid pupil samples**: Non-finite values are excluded before interpolation (handled by `interpolate_pupil`).
- **Few trials**: Sessions with fewer than 2 valid trials are skipped.
- **Invalid ROIs**: If `valid_roi` flag differs from event shape, invalid ROIs are filtered out.

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
if int(raw["keep_mask"].sum()) < 2:
    ...continue
...
valid = np.isfinite(pupil_diameter)
```

iii. The AI's handling of missing data is similar to the reference. The reference uses try/except for failed sessions and maps NaN to bin 0; the AI excludes sessions missing eye tracking upfront and interpolates through NaN gaps.

## 9-a. What are the most time-consuming steps of the code?

i. The AI identified that neural interpolation is the dominant cost: "Neural interpolation is still the dominant expected cost because every kept trial needs event traces resampled onto the common grid." Loading NWB files via h5py is also time-consuming.

ii. N/A

iii. The two-pass design (pass 1 collects global statistics without neural arrays, pass 2 does full conversion) helps manage memory. Full conversion ran in ~356 seconds for 199 sessions.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_sessions` iterates over each trial within a session to compute interpolated signals and output codes. The stimulus identity/change code functions use `np.searchsorted` and vectorized operations, but the outer trial loop remains sequential.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
    neural_trial = interpolate_matrix(...)
    ...
```

iii. The AI noted that trial-level interpolation is vectorized across neurons, but the per-trial loop could theoretically be batched.

## 9-c. What processing does the code repeat multiple times?

i. The code uses a two-pass design: pass 1 loads each session to compute global statistics (running/pupil bin edges, image vocabulary), then pass 2 re-loads each session for full conversion. This means every NWB file is read twice.

ii.
```python
# Pass 1
running_edges, pupil_edges, image_values, valid_counts, native_dt = collect_global_statistics(sessions)
# Pass 2
data = convert_sessions(sessions=sessions, ...)
```

iii. The AI acknowledged this but justified it as a memory optimization: "Two-pass design avoids storing all neural arrays while computing global bin edges." The reference also uses a similar two-pass approach (collect continuous values first, then discretize).

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI rebins all data from native ophys rate to 30 Hz, which introduces interpolation artifacts in the neural data. This is extra processing not present in the reference. Additionally, the stimulus presentations table is loaded and processed to derive image identity and change codes, which is more complex than the reference's simpler trial-table approach.

ii. N/A

iii. The 30 Hz resampling adds computational cost and potential signal distortion. The reference's approach of keeping native ophys frames is simpler and avoids this.
