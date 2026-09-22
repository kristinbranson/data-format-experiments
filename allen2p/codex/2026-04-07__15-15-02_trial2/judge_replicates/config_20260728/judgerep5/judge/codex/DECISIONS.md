# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script enumerates local NWB files by joining `ophys_experiment_table.csv` to the filenames under `behavior_ophys_experiments/`, drops passive experiments, removes active experiments that lack pupil-tracking groups, and then reads each kept NWB directly with `h5py`. It does not use the Allen SDK cache object and it does not filter `project_code == "VisualBehavior"`, so locally present `VisualBehaviorMultiscope` experiments can be included.

ii.
```python
exp_table = pd.read_csv(METADATA_DIR / "ophys_experiment_table.csv")
file_map = {
    int(path.stem.split("_")[-1]): path
    for path in sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))
}
exp_table = exp_table[exp_table["ophys_experiment_id"].isin(file_map)].copy()
exp_table = exp_table[~exp_table["passive"]].copy()
...
with h5py.File(session.path, "r") as h5f:
```

iii. In `CONVERSION_NOTES.md` Step 5 and Step 6, the agent justified direct HDF5 reads as a workaround for a `pynwb/hdmf` incompatibility with the provided NWB files, and justified restricting to active sessions because the decoder requires trial outcomes and usable pupil signals.

## 1-b. How are the data split into subjects?

i. Subjects are defined by unique `mouse_id` values from the experiment metadata, converted to strings and indexed in first-seen order as sessions are assembled.

ii.
```python
mouse_id=str(int(row.mouse_id))
...
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
```

iii. The notes state that mouse identifiers from the metadata should become `subjects` / `subject_idx`, so the script uses the metadata `mouse_id` as the canonical subject split.

## 1-c. How are the data split into sessions?

i. Each NWB experiment file is treated as one decoder session. The code builds one `SessionInfo` per `ophys_experiment_id`; it does not regroup experiments that share an `ophys_session_id`.

ii.
```python
sessions = [
    SessionInfo(
        ophys_experiment_id=int(row.ophys_experiment_id),
        path=file_map[int(row.ophys_experiment_id)],
        mouse_id=str(int(row.mouse_id)),
        ...
    )
    for row in exp_table.itertuples(index=False)
]
```

iii. In Step 4 of `CONVERSION_NOTES.md`, the agent explicitly resolved the session-definition discrepancy by treating each NWB experiment file as one decoder session because the local files are experiment-specific and contain experiment-specific neural traces.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table. For each kept trial, the code uses the trial `start_time` and `stop_time` and builds a variable-length 30 Hz grid spanning that interval.

ii.
```python
trial_group = h5f["intervals"]["trials"]
trials = read_interval_table(trial_group, [..., "start_time", "stop_time"])
...
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. The agent’s notes say trial boundaries should come directly from the NWB / SDK trials table and run from `start_time` to `stop_time`, because that preserves the experiment’s behavioral trial definition while still allowing time-varying outputs.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if they are `go` or `catch` and are not `aborted` and not `auto_rewarded`. Sessions with fewer than two kept trials are dropped. There is no explicit `change_time.notna()` filter and no explicit empty-window check.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
...
if int(raw["keep_mask"].sum()) < 2:
    print(f"[pass2] skipping session {session.ophys_experiment_id} because it has "
          f"{int(raw['keep_mask'].sum())} valid trials")
    continue
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the agent justified this as following the common trial taxonomy across the SDK, whitepaper, and paper: include `go` and `catch`, exclude `aborted` and `auto_rewarded`, and require at least two valid trials per kept session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the NWB event-detection output, specifically `processing/ophys/event_detection/data` together with the matching event timestamps.

ii.
```python
ophys_timestamps = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["timestamps"], dtype=np.float64
)
events = np.asarray(
    h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32
)
```

iii. The agent’s Step 4 and Step 5 notes say it chose precomputed calcium events rather than dF/F because the strategy paper’s analyses used detected calcium events, and because those events are already present in the released NWB files.

## 2-b. How is the `neural` data processed?

i. The event matrix is linearly interpolated from its native ophys timestamps onto each trial’s common 30 Hz grid, then transposed to `(n_neurons, n_timepoints)` and stored as `float32`.

ii.
```python
def interpolate_matrix(source_t, source_values, query_t):
    ...
    out = left_vals * (1.0 - w[:, None]) + right_vals * w[:, None]
    return out.astype(np.float32)
...
neural_trial = interpolate_matrix(
    raw["ophys_timestamps"], raw["events"], grid
).T.astype(np.float32)
```

iii. In the trajectory and Step 5 notes, the agent justified a common 30 Hz grid because the target format needs one bin size across sessions and the paper interpolates event-triggered signals onto common 30 Hz timestamps.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code conditionally filters neurons using the NWB `valid_roi` flag from the cell-specimen table if the event matrix still contains the full unfiltered ROI set. Otherwise it keeps the events as stored. No additional activity-based neuron filtering is applied.

ii.
```python
cell_table = h5f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. Step 4 of `CONVERSION_NOTES.md` says the agent wanted to preserve the release’s existing ROI QC and avoid adding ad hoc neuron curation beyond the reference filtering already encoded in the files.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to trial start. For each trial, the time axis starts at the trial’s `start_time` and extends to `stop_time`, and the neural event trace is sampled on that trial-local grid.

ii.
```python
def session_grid(start: float, stop: float, dt: float = TIME_BIN_SIZE_S) -> np.ndarray:
    n_bins = max(1, int(math.ceil((stop - start) / dt)))
    return start + np.arange(n_bins, dtype=np.float64) * dt
...
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. The notes say trial segmentation should follow `start_time` to `stop_time`; the saved metadata also records `temporal_alignment_event` as `"trial start"`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a fixed 30 Hz grid, i.e. `1/30` s or `33.33` ms per bin. Yes, temporal resampling is applied to every stream, including the neural data.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
TIME_BIN_SIZE_MS = TIME_BIN_SIZE_S * 1000.0
...
"time_bin_size": float(TIME_BIN_SIZE_MS),
```

iii. The Step 5 notes explicitly justify 30 Hz as the common bin size because behavior and eye tracking are naturally near 30 Hz and the local dataset mixes different native ophys frame rates.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from the chosen stimulus-presentation table, using presentation `start_time`, `stop_time`, `image_name`, and `omitted`. The code does not use `initial_image_name` / `change_image_name` from the trials table.

ii.
```python
stim = read_interval_table(
    stim_group,
    ["start_time", "stop_time", "image_name", "is_change", "omitted", ...],
)
...
starts = stimulus["start_time"]
stops = stimulus["stop_time"]
image_names = stimulus["image_name"]
omitted = stimulus["omitted"]
```

iii. The mapping notes say the agent preferred the stimulus-presentation table so that image identity would reflect the actual flashed-image stream, including gray gaps and omitted flashes.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each trial bin, the code finds the most recent presentation interval whose `start_time` precedes the bin, checks whether the bin falls before that interval’s `stop_time`, and then assigns either that presentation’s image code or a special `"gray"` code. Omissions and inter-stimulus gaps are labeled `"gray"`. A global codebook is built from all non-omitted image names plus `"gray"`.

ii.
```python
image_values = ["gray"] + sorted(image_names)
...
idx = np.searchsorted(starts, query_t, side="right") - 1
codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
...
if (not is_omitted) and str(name) in image_to_code:
    target[i] = image_to_code[str(name)]
```

iii. Step 5 of `CONVERSION_NOTES.md` says the agent intentionally added `"gray"` as an explicit class because the task contains 500 ms gray periods and omission periods, and it wanted a complete time-varying label.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Image identity is evaluated on the same per-trial 30 Hz grid that is used for the neural interpolation, so both are aligned bin-for-bin.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
```

iii. The agent’s notes describe a single common trial grid for neural and behavioral/stimulus outputs so that every output can be decoded from the same aligned bins.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change is derived from the stimulus-presentation table’s `is_change`, `start_time`, `stop_time`, and `omitted` columns. It is not derived from trial `change_time` plus `go`.

ii.
```python
starts = stimulus["start_time"]
stops = stimulus["stop_time"]
is_change = stimulus["is_change"]
omitted = stimulus["omitted"]
```

iii. In the trajectory and notes, the agent justified using `is_change` intervals because they describe the actual changed-image presentation window and yield a less degenerate decoder target than a single-bin impulse.

## 4-b. What processing is involved in computing `output` *Image change*?

i. The code locates the stimulus-presentation interval covering each trial bin and assigns `1` if that interval is a non-omitted change presentation and `0` otherwise.

ii.
```python
idx = np.searchsorted(starts, query_t, side="right") - 1
codes = np.zeros(query_t.shape, dtype=np.int64)
...
changed = is_change[sub_idx] & (~omitted[sub_idx])
codes[assign] = changed.astype(np.int64)
```

iii. The agent recorded that this was an intentional change late in development to better match task semantics and improve the usefulness of the decoder target.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No continuous thresholding is applied. The output is directly encoded as the binary categories `0 = no_change` and `1 = change`.

ii.
```python
"output_values": [
    list(image_values),
    ["no_change", "change"],
    ...
]
...
codes = np.zeros(query_t.shape, dtype=np.int64)
codes[assign] = changed.astype(np.int64)
```

iii. The agent treated image change as intrinsically binary because the raw variable is already a boolean task state.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Like image identity, image change is evaluated on the same per-trial 30 Hz grid used for the neural data, so alignment is binwise.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. The common-grid strategy in the notes applies to all decoder outputs, including the binary image-change label.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed comes from the NWB running-speed dataset and its timestamps: `processing/running/speed/data` and `processing/running/speed/timestamps`.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"], dtype=np.float32)
running_timestamps = np.asarray(
    h5f["processing"]["running"]["speed"]["timestamps"], dtype=np.float64
)
```

iii. The notes say the agent wanted to use the release’s precomputed running-speed signal rather than recomputing it, consistent with the SDK/whitepaper processing pipeline.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is linearly interpolated onto each trial’s 30 Hz grid, pooled across all sessions to compute global quintile edges, and then digitized into five bins.

ii.
```python
running_values.append(
    interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
)
...
running_edges = robust_quintile_edges(running_all)
...
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Step 5 of `CONVERSION_NOTES.md` says continuous outputs should share global class definitions, so the agent used dataset-wide quintile bins rather than per-session bins.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. The script computes the 20th, 40th, 60th, and 80th percentiles across all interpolated running-speed samples, forces strictly increasing edges if ties occur, and then assigns bin indices `0` through `4` with `np.digitize`.

ii.
```python
def robust_quintile_edges(values: np.ndarray) -> np.ndarray:
    percentiles = np.nanpercentile(values, [20, 40, 60, 80]).astype(np.float64)
    for i in range(1, len(percentiles)):
        if percentiles[i] <= percentiles[i - 1]:
            percentiles[i] = percentiles[i - 1] + 1e-6
    return percentiles
...
return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. The agent’s justification was robustness: quintiles implement the requested five equal-percentile bins, and the epsilon adjustment prevents degenerate repeated edges.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is interpolated onto the same trial grid used for the neural events, then binned on that same grid.

ii.
```python
running_cont = interpolate_vector(
    raw["running_timestamps"], raw["running_speed"], grid
)
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. The notes explicitly describe the 30 Hz common grid as the shared alignment reference for neural, running, pupil, and stimulus labels.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. Pupil diameter is derived from the raw pupil-tracking ellipse width and height arrays plus eye-tracking timestamps. The code defines diameter as `max(width, height)`.

ii.
```python
pupil_width = np.asarray(
    h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["width"], dtype=np.float32
)
pupil_height = np.asarray(
    h5f["acquisition"]["EyeTracking"]["pupil_tracking"]["height"], dtype=np.float32
)
pupil_timestamps = np.asarray(
    h5f["acquisition"]["EyeTracking"]["eye_tracking"]["timestamps"], dtype=np.float64
)
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```

iii. In Step 5, the agent justified this as a direct pupil-diameter proxy available in the NWB file without having to reconstruct a separate area-to-diameter transform.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The script keeps only finite pupil-diameter samples, linearly interpolates them onto each trial’s 30 Hz grid, pools all interpolated values to compute global quintile edges, and digitizes each trial into five bins. It does not use the SDK blink mask.

ii.
```python
def interpolate_pupil(pupil_t, pupil_diameter, query_t):
    valid = np.isfinite(pupil_diameter)
    ...
    return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
...
pupil_values.append(
    interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
)
...
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. The notes justify global quintile discretization the same way as running speed. The agent’s broader justification for this pathway was that usable pupil data were required for the decoder, which is why sessions missing the pupil-tracking groups were excluded earlier.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. The script uses global quintile binning, exactly like running speed: 20/40/60/80 percentile edges, then `np.digitize` into bins `0` through `4`.

ii.
```python
pupil_edges = robust_quintile_edges(pupil_all)
...
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. The agent used the same global-bin policy for both continuous behavioral outputs so that their categorical levels would be consistent across sessions.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto the same per-trial 30 Hz grid used for the neural events, then binned on that aligned grid.

ii.
```python
pupil_cont = interpolate_pupil(
    raw["pupil_timestamps"], raw["pupil_diameter"], grid
)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. As in the notes, all decoder outputs share the same common trial grid, so alignment is handled the same way for pupil as for running speed.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Trial outcome is derived from the boolean trial columns `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
def trial_outcome_code(trials: Dict[str, np.ndarray], idx: int, mapping: Dict[str, int]) -> int:
    for name in ("hit", "miss", "false_alarm", "correct_reject"):
        if bool(trials[name][idx]):
            return mapping[name]
```

iii. The notes describe these as the canonical outcome categories from the SDK / NWB trial taxonomy for active go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The code maps those four mutually exclusive labels to integers and broadcasts the chosen code across all time bins of the trial.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_to_code = {name: idx for idx, name in enumerate(outcome_values)}
...
outcome_code = trial_outcome_code(raw["trials"], trial_idx, outcome_to_code)
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. In Step 5, the agent explicitly decided to represent all outputs as time-varying for decoder convenience, even when the underlying variable is static per trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles several edge cases by exclusion or fallback: sessions missing the pupil-tracking groups are dropped up front; sessions with fewer than two valid trials are skipped; pupil interpolation ignores non-finite samples and falls back to a constant trace if only one valid sample exists; if no valid pupil sample exists it raises an error; tied quintile edges are perturbed by `1e-6`; and neuron filtering uses `valid_roi` when needed. Missing data are generally not preserved as NaNs because `np.interp` fills endpoint values.

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
...
if valid.sum() == 0:
    raise ValueError("No valid pupil samples available")
if valid.sum() == 1:
    return np.full(query_t.shape, float(pupil_diameter[valid][0]), dtype=np.float32)
...
if percentiles[i] <= percentiles[i - 1]:
    percentiles[i] = percentiles[i - 1] + 1e-6
```

iii. The notes frame these choices as pragmatic validation-oriented handling: skip unusable sessions, keep decoder labels well-formed, and avoid adding extra custom curation beyond the release QC already present in the files.

## 9-a. What are the most time-consuming steps of the code?

i. The dominant costs are the two full passes over all sessions and the per-trial interpolation work, especially interpolating the neural event matrix for every kept trial. Session I/O is also substantial because each pass reopens every NWB file.

ii.
```python
for idx, session in enumerate(sessions, start=1):
    raw = read_session_raw(session, load_events=False)
    ...
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
    ...
    neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. Step 6 of `CONVERSION_NOTES.md` explicitly says neural interpolation is the dominant expected cost and that the two-pass structure was chosen as a memory/runtime tradeoff.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The code contains several obvious Python-level loops that could be reduced: iterating trial-by-trial in both passes, iterating sample-by-sample inside `stimulus_identity_codes`, and reopening / reprocessing every session twice instead of fusing work.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    ...
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    if (not is_omitted) and str(name) in image_to_code:
        target[i] = image_to_code[str(name)]
```

iii. The agent’s notes acknowledge that neural interpolation remains the bottleneck and emphasize vectorization only within each trial; the remaining outer loops were left in place for implementation simplicity.

## 9-c. What processing does the code repeat multiple times?

i. The code rereads every session twice, rebuilds the same per-trial time grids twice, and re-interpolates running speed and pupil traces once in pass 1 for global statistics and again in pass 2 for final outputs.

ii.
```python
raw = read_session_raw(session, load_events=False)
...
grid = session_grid(start, stop)
running_values.append(interpolate_vector(..., grid))
pupil_values.append(interpolate_pupil(..., grid))
...
raw = read_session_raw(session)
...
grid = session_grid(start, stop)
running_cont = interpolate_vector(..., grid)
pupil_cont = interpolate_pupil(..., grid)
```

iii. Step 6 of `CONVERSION_NOTES.md` justifies this repetition as the price of a two-pass design that avoids storing all trial-level neural arrays in memory while computing global bin edges.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several values are computed only for bookkeeping or optional diagnostics: `valid_trial_counts` and `native_dt_by_session` are only printed, the stimulus fields `trials_id`, `active`, and `flashes_since_change` are loaded but not used in the final dataset, and optional processing plots are generated outside the saved pickle. The declared `RNG` is unused.

ii.
```python
valid_trial_counts[session.ophys_experiment_id] = int(raw["keep_mask"].sum())
native_dt_by_session[session.ophys_experiment_id] = session_native_dt(raw["ophys_timestamps"])
...
stim = read_interval_table(
    stim_group,
    ["start_time", "stop_time", "image_name", "is_change", "omitted",
     "trials_id", "active", "flashes_since_change"],
)
...
RNG = np.random.default_rng(0)
```

iii. The notes justify the extra bookkeeping as sanity checking and validation support, but those values do not contribute to the final decoder dataset.
