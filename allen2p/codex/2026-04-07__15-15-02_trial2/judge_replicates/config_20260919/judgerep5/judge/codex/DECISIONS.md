# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the local release directly with `h5py`. It joins NWB filenames to `ophys_experiment_table.csv`, removes passive experiments, and removes files without the required eye-tracking groups. Each remaining NWB is processed twice: once for global category/bin statistics and once for final conversion.

ii.
```python
exp_table = pd.read_csv(METADATA_DIR / "ophys_experiment_table.csv")
file_map = {int(path.stem.split("_")[-1]): path
            for path in sorted(EXPERIMENT_DIR.glob("behavior_ophys_experiment_*.nwb"))}
exp_table = exp_table[exp_table["ophys_experiment_id"].isin(file_map)].copy()
exp_table = exp_table[~exp_table["passive"]].copy()
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
```

iii. The notes say the local files are a curated subset and direct HDF5 was chosen because the installed `pynwb/hdmf` stack could not instantiate the NWBs. Active sessions were selected because trial outcome is required; sessions lacking pupil data were excluded.

## 1-b. How are the data split into subjects?

i. A subject is the metadata-table `mouse_id`. Unique string IDs are registered when an included experiment is converted, and `subject_idx` points from each converted experiment to that list.

ii.
```python
mouse_id=str(int(row.mouse_id))
subject_idx = subject_to_idx.setdefault(session.mouse_id, len(subject_to_idx))
if subject_idx == len(data["subjects"]):
    data["subjects"].append(session.mouse_id)
```

iii. The agent justified this as the SDK/release animal identifier and reported 38 mice in the usable local subset.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id`/NWB file is one decoder session; experiments sharing an `ophys_session_id` are not combined.

ii.
```python
class SessionInfo:
    ophys_experiment_id: int
    path: Path
...
for sess_num, session in enumerate(sessions, start=1):
    raw = read_session_raw(session)
    data["neural"].append(session_neural)
```

iii. The notes explicitly acknowledge that there are fewer unique ophys sessions than experiment files, but choose experiment files because traces are experiment-specific and one file supplies a coherent neuron population.

## 1-d. How are the data split into trials?

i. Trial rows come from `intervals/trials`. Each retained row is represented from `start_time` (inclusive grid origin) to `stop_time` on a variable-length 30 Hz grid.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    start = float(raw["trials"]["start_time"][trial_idx])
    stop = float(raw["trials"]["stop_time"][trial_idx])
    grid = session_grid(start, stop)
```

iii. The trial table was treated as the canonical task definition. Full trial windows preserve pre-change and post-change behavior and allow time-varying labels.

## 1-e. How are trials filtered based on quality controls?

i. Only go or catch trials are retained; aborted and auto-rewarded trials are removed. Converted sessions must have at least two retained trials.

ii.
```python
keep = (trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])
if int(raw["keep_mask"].sum()) < 2:
    continue
```

iii. This follows the explicit task instruction. The two-trial minimum is required by the decoder validator.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is the NWB precomputed calcium-event matrix and its event-detection timestamps.

ii.
```python
ophys_timestamps = np.asarray(h5f["processing"]["ophys"]["event_detection"]["timestamps"])
events = np.asarray(h5f["processing"]["ophys"]["event_detection"]["data"], dtype=np.float32)
```

iii. The agent chose events because the paper says analyses used detected calcium events, even though dF/F is also available.

## 2-b. How is the `neural` data processed?

i. Time-by-cell events are linearly interpolated onto every trial's 30 Hz grid and transposed to neuron-by-time. No normalization is added.

ii.
```python
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T.astype(np.float32)
```

iii. The notes cite the paper's interpolation to common 30 Hz timestamps and say vectorizing interpolation over neurons controls runtime.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code conditionally applies the cell table's `valid_roi` mask only when the event matrix width equals the unfiltered cell-table length and the mask would reduce it. It adds no other cell filter.

ii.
```python
if events.shape[1] == n_cell_table and "valid_roi" in cell_table:
    valid_roi = np.asarray(h5_array(cell_table["valid_roi"])).astype(bool)
    if valid_roi.sum() != events.shape[1]:
        events = events[:, valid_roi]
```

iii. The agent says released NWBs already reflect reference ROI QC and that ad hoc extra filtering would diverge from the reference pipeline.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are aligned to trial start: the trial grid starts at `start_time`, ends before `stop_time`, and all streams are evaluated at the identical absolute timestamps.

ii.
```python
grid = session_grid(start, stop)
neural_trial = interpolate_matrix(raw["ophys_timestamps"], raw["events"], grid).T
```

iii. The notes describe synchronized absolute experiment clocks and record `temporal_alignment_event` as trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has fixed 33.333 ms bins (30 Hz). Neural traces from native rates near 11 or 31 Hz are linearly resampled; this is interpolation, not aggregation.

ii.
```python
TIME_BIN_SIZE_S = 1.0 / 30.0
n_bins = max(1, int(math.ceil((stop - start) / dt)))
return start + np.arange(n_bins) * dt
```

iii. A common bin size was chosen to satisfy the shared-bin requirement across mixed-rate sessions and to match the agent's reading of the strategy paper.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It uses the active stimulus-presentation table's `start_time`, `stop_time`, `image_name`, and `omitted` fields.

ii.
```python
stim = read_interval_table(stim_group,
    ["start_time", "stop_time", "image_name", "is_change", "omitted", ...])
```

iii. The agent preferred actual presentation intervals over trial-level initial/change names so gray gaps and omissions are represented.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global vocabulary is `gray` plus sorted nonempty, non-omitted image names. At each query time the most recent presentation is found; its image code is used only while within its interval, otherwise gray is emitted.

ii.
```python
image_values = ["gray"] + sorted(image_names)
idx = np.searchsorted(starts, query_t, side="right") - 1
codes = np.full(query_t.shape, image_to_code["gray"], dtype=np.int64)
in_interval = query_t[valid] < stops[idx_valid]
```

iii. The rationale is that image identity is requested only during the non-gray screen and gray/omission periods need an explicit class for a complete time series.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is evaluated at the exact same 30 Hz `grid` used to interpolate neural activity.

ii.
```python
image_codes = stimulus_identity_codes(raw["stimulus"], grid, image_to_code)
```

iii. Shared absolute query timestamps were used to avoid cross-stream offsets.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It is derived from stimulus-presentation `is_change`, presentation start/stop times, and `omitted`.

ii.
```python
is_change = stimulus["is_change"]
omitted = stimulus["omitted"]
changed = is_change[sub_idx] & (~omitted[sub_idx])
```

iii. The agent says this marks actual changed-image presentations and leaves catch trials at zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each grid point, the code locates its presentation interval and assigns that presentation's change flag; points outside presentations are zero.

ii.
```python
idx = np.searchsorted(starts, query_t, side="right") - 1
codes = np.zeros(query_t.shape, dtype=np.int64)
codes[assign] = changed.astype(np.int64)
```

iii. The original one-bin impulse was changed to the changed-image presentation window because the agent considered it less degenerate while still task-aligned.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No numeric threshold is used. Boolean `is_change & ~omitted` maps directly to 1; everything else maps to 0, named `change` and `no_change`.

ii.
```python
"output_values": [..., ["no_change", "change"], ...]
```

iii. The source flag is already categorical, so further thresholding was unnecessary.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change labels are evaluated on the same trial grid as neural interpolation and remain 1 for the changed-image presentation interval.

ii.
```python
image_change = stimulus_change_codes(raw["stimulus"], grid)
```

iii. The common grid was intended to provide exact binwise correspondence.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `processing/running/speed/data` and its timestamps.

ii.
```python
running_speed = np.asarray(h5f["processing"]["running"]["speed"]["data"])
running_timestamps = np.asarray(h5f["processing"]["running"]["speed"]["timestamps"])
```

iii. This is the released/SDK-standard filtered running-speed stream.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Speed is linearly interpolated to each 30 Hz trial grid. Four global 20/40/60/80 percentile edges are computed over all retained trial bins in pass 1 and applied in pass 2.

ii.
```python
running_values.append(interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid))
running_edges = robust_quintile_edges(running_all)
running_bins = digitize_with_edges(running_cont, running_edges)
```

iii. Global quintiles give shared categories and approximately balanced classes across the dataset.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize(..., right=False)` converts values into bins 0-4 at global quintile cut points; duplicate edges are nudged upward by `1e-6`.

ii.
```python
percentiles = np.nanpercentile(values, [20, 40, 60, 80])
return np.digitize(values, edges, right=False).astype(np.int64)
```

iii. Equal-percentile categories were required by the task and improve class balance.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed and neural activity are independently interpolated at the same absolute 30 Hz trial timestamps.

ii.
```python
running_cont = interpolate_vector(raw["running_timestamps"], raw["running_speed"], grid)
```

iii. The streams are hardware-synchronized, so common-time interpolation was deemed valid.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. The code reads eye-tracking pupil `width` and `height` plus eye-tracking timestamps, then defines diameter as their maximum. It does not read or apply a blink flag.

ii.
```python
pupil_width = np.asarray(...["pupil_tracking"]["width"])
pupil_height = np.asarray(...["pupil_tracking"]["height"])
pupil_diameter = np.maximum(pupil_width, pupil_height).astype(np.float32)
```

iii. The notes describe `max(width, height)` as diameter and repeatedly claim blink-masked values, but that masking is absent from the implementation.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Nonfinite diameter samples are discarded for interpolation; remaining samples are linearly interpolated to each 30 Hz grid. Global quintile edges are computed and applied just like running speed.

ii.
```python
valid = np.isfinite(pupil_diameter)
return np.interp(query_t, pupil_t[valid], pupil_diameter[valid]).astype(np.float32)
pupil_edges = robust_quintile_edges(pupil_all)
```

iii. The stated goal was to fill small invalid/blink gaps by interpolation and create globally consistent balanced categories, although explicit blink masking was not implemented.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Global 20/40/60/80 percentile cut points define integer bins 0-4, with non-increasing cut points nudged apart.

ii.
```python
pupil_bins = digitize_with_edges(pupil_cont, pupil_edges)
```

iii. This directly follows the requested five equal-percentile bins.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated at the identical 30 Hz trial timestamps used for neural activity.

ii.
```python
pupil_cont = interpolate_pupil(raw["pupil_timestamps"], raw["pupil_diameter"], grid)
```

iii. The agent relied on synchronized clocks and confirmed selected raw-to-converted reconstructions with `np.allclose`.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome comes from the trials table's mutually exclusive `hit`, `miss`, `false_alarm`, and `correct_reject` booleans.

ii.
```python
for name in ("hit", "miss", "false_alarm", "correct_reject"):
    if bool(trials[name][idx]):
        return mapping[name]
```

iii. These are the canonical outcomes for retained go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The four outcomes map to codes 0-3 in fixed order. The selected code is broadcast across every time bin of the trial; missing/ambiguous outcome labels raise an error.

ii.
```python
outcome_values = ["hit", "miss", "false_alarm", "correct_reject"]
trial_outcome = np.full(grid.shape, outcome_code, dtype=np.int64)
```

iii. Broadcasting makes the static target compatible with the decoder's time-varying output matrix.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Experiments without required eye tracking are excluded. Nonfinite pupil values are interpolated over; zero or one valid pupil sample raises or becomes a constant, respectively. Interpolation uses `np.interp`, so queries beyond endpoints take endpoint values. Sessions under two trials are skipped. Missing required tables/columns or missing outcome labels raise errors rather than being silently repaired.

ii.
```python
filtered_sessions = [session for session in sessions if has_required_eye_tracking(session.path)]
if valid.sum() == 0: raise ValueError("No valid pupil samples available")
if valid.sum() == 1: return np.full(query_t.shape, float(pupil_diameter[valid][0]))
```

iii. The agent excluded three eye-tracking-free sessions after a failed full run and regarded interpolation as appropriate for small gaps. Sparse all-zero event trials were retained after raw-data verification.

## 9-a. What are the most time-consuming steps of the code?

i. Reading NWBs twice and, especially, interpolating every retained trial's full time-by-neuron event matrix onto 30 Hz dominate runtime.

ii.
```python
raw = read_session_raw(session, load_events=False)  # pass 1
raw = read_session_raw(session)                     # pass 2
neural_trial = interpolate_matrix(...).T
```

iii. The notes predicted neural interpolation as the dominant cost and measured about 356 seconds for the full local conversion.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Sessions and trials are Python loops. The per-presentation image-name loop and repeated per-trial interpolation could be further vectorized or performed once per session; interpolation across neurons is already vectorized.

ii.
```python
for trial_idx in np.flatnonzero(raw["keep_mask"]):
    ...
for i, (name, is_omitted) in enumerate(zip(names, omit)):
    ...
```

iii. The agent explicitly optimized the expensive neuron dimension but kept trial loops for variable-length boundaries and clarity.

## 9-c. What processing does the code repeat multiple times?

i. Every included file is opened in both passes. Trial grids and running/pupil interpolation are recomputed in both passes; pass 2 additionally reads events. Stimulus lookup is also performed trial by trial.

ii.
```python
collect_global_statistics(sessions)
convert_sessions(sessions=..., running_edges=..., pupil_edges=...)
```

iii. This two-pass design was intentional to keep memory bounded while deriving global bin edges and vocabulary.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Pass 1 calculates and returns per-session native frame intervals and valid-trial counts only for logging. It also scans stimulus columns such as `trials_id` and `flashes_since_change` that are never used in conversion. Continuous running/pupil arrays are recomputed and discarded after categorization, and optional plots add work when enabled.

ii.
```python
stim = read_interval_table(stim_group, [..., "trials_id", "active", "flashes_since_change"])
return running_edges, pupil_edges, image_values, valid_trial_counts, native_dt_by_session
```

iii. These values support diagnostics, sanity checks, and progress reporting, but they do not enter the saved decoder arrays.
