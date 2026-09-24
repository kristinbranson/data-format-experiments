# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK `VisualBehaviorOphysProjectCache`. It reads the local Visual Behavior Ophys cache directly: the CSV manifest `project_metadata/ophys_experiment_table.csv` is loaded with pandas, intersected with the NWB files actually present on disk (`behavior_ophys_experiment_<ophys_experiment_id>.nwb`, 284 files), then filtered to `passive == False` (202 experiments) and sorted by `ophys_experiment_id`. Each retained row becomes a `SessionMeta` record. Every data stream (neural events, running, eye tracking, trials table, stimulus presentation tables, cell-specimen table) is then read out of the NWB/HDF5 file with `h5py` in two passes over the file list (pass 1 = global quantile edges + session QC, pass 2 = conversion). No filter on `project_code` is applied, so the retained set is 168 `VisualBehavior` + 34 `VisualBehaviorMultiscope` experiments.

ii.
```python
def get_local_session_metadata(data_root: Path) -> list[SessionMeta]:
    table_path = data_root / "visual-behavior-ophys-1.1.0" / "project_metadata" / "ophys_experiment_table.csv"
    exp_table = pd.read_csv(table_path)

    experiment_dir = data_root / "visual-behavior-ophys-1.1.0" / "behavior_ophys_experiments"
    available_files = {
        int(path.stem.split("_")[-1]): path
        for path in sorted(experiment_dir.glob("behavior_ophys_experiment_*.nwb"))
    }

    exp_table = exp_table[exp_table["ophys_experiment_id"].isin(available_files)].copy()
    exp_table = exp_table[~exp_table["passive"]].copy()
    exp_table = exp_table.sort_values("ophys_experiment_id")
```
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```
```python
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)  # pass 1
...
for idx, session in enumerate(kept_sessions, start=1):                                        # pass 2
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. From CONVERSION_NOTES.md Step 4/Step 6: "Local environment cannot instantiate `NWBFile` for these NWBs due `external_resources`/version mismatch … Read NWB files directly with `h5py` and mirror the AllenSDK/whitepaper semantics from the processed NWB contents rather than relying on broken high-level loading in this environment." The AI notes that the NWB files already contain the Allen-processed products (dF/F, event detection, trials table, stimulus presentations, blink-filtered eye tracking), so reading them directly uses the same processed sources as the SDK. Passive experiments are excluded because "Passive sessions are not task performance and make outcome labels degenerate" (Step 4). The trajectory shows the AI enumerated alternative scopes (active only, familiar only, multiscope only, VISp/VISl only) and settled on "all locally available active experiment NWBs" (Key Decision 1).

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values (as strings) taken from `ophys_experiment_table.csv` for the sessions that survive pass-1 QC. `subjects` is the sorted unique list; `subject_idx[i]` is the index of the mouse that produced converted session `i`. Result: 38 mice.

ii.
```python
mouse_id=str(row.mouse_id),
...
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. CONVERSION_NOTES.md Step 5 variable-mapping table: "`mouse_id` from `ophys_experiment_table.csv` → `subjects`, `subject_idx`; Unique string list + per-session index". `mouse_id` is the Allen release's canonical animal identifier; the AI cross-checked the converted subject count (38) against a direct recount from the raw metadata in Step 10 Check 4.

## 1-c. How are the data split into sessions?

i. **One converted "session" = one `ophys_experiment_id` (one imaging plane)**, not one `ophys_session_id`. Sessions are ordered by `ophys_experiment_id`. Because the retained set includes 34 `VisualBehaviorMultiscope` planes (all from mouse 457841, 6 real ophys sessions, up to 7 planes each), those simultaneously recorded planes are emitted as up to 7 separate "sessions" that share the same behavioral session, the same trials and identical output traces. Sessions dropped: 3 for missing eye tracking, and any with fewer than 2 usable trials — 199 sessions kept out of 202.

ii.
```python
sessions.append(SessionMeta(
    ophys_experiment_id=int(row.ophys_experiment_id),
    ophys_session_id=int(row.ophys_session_id),
    behavior_session_id=int(row.behavior_session_id), ...))
...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(session=session, ...)
    neural_all.append(neural_trials)
```
```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

iii. Key Decision 2: "Treat each `ophys_experiment_id` file as one converted session: This matches the AllenSDK object granularity (`BehaviorOphysExperiment`) and yields a single imaging plane / neuron set / brain region per session." Step 10 Check 5 explicitly acknowledges the consequence: "Multiple experiment files can share the same `behavior_session_id` or `ophys_session_id`; the conversion intentionally treats each `ophys_experiment_id` plane as a separate session because that is the AllenSDK experiment granularity and each file has its own neuron set." The `brain_regions`/`brain_region_idx` design follows from this (one `targeted_structure` per session): regions are `VISp` (29,006 neurons) and `VISl` (162 neurons).

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table (the Allen-processed trial log). The kept set is `(go | catch) & ~aborted & ~auto_rewarded`, sorted by trial `id`. Each trial spans `start_time → stop_time` (variable length, mean ≈ 254 bins ≈ 8.5 s at 30 Hz), sampled on a uniform 30 Hz grid of bin centers anchored at the trial start. Total: 51,075 trials over 199 sessions (mean 256.7/session).

ii.
```python
trials = get_trial_table(f)
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
trials = trials.sort_values("id").reset_index(drop=True)
```
```python
def build_trial_bins(start_time: float, stop_time: float) -> np.ndarray:
    if not np.isfinite(start_time) or not np.isfinite(stop_time) or stop_time <= start_time:
        return np.asarray([], dtype=np.float64)
    n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
    centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
    valid = centers < (stop_time + 1e-9)
    return centers[valid]
```

iii. Key Decisions 3–4 and Step 1 notes: the NWB trials table is the Allen-processed trial definition ("mirrors the Allen reference processing without re-deriving trial logic from lower-level files"), and `trial_masks.contingent_trials` in the SDK "explicitly defines contingent trials as GO and CATCH only", which the AI matched to the instruction to include Go and Catch and exclude Aborted and Auto-rewarded. It used the full `start_time → stop_time` window so the trial contains the pre-change flashes and the post-change response period, enabling time-varying outputs.

## 1-e. How are trials filtered based on quality controls?

i. Filters applied: (a) trial-type filter `(go|catch) & ~aborted & ~auto_rewarded`; (b) trials whose bin grid is empty (non-finite or non-positive `stop_time - start_time`) are skipped; (c) sessions with fewer than 2 kept trials are dropped in pass 1, and `convert_session` raises if fewer than 2 usable trials remain; (d) whole sessions lacking an `EyeTracking` acquisition group are dropped (3 sessions), since pupil is a required output. Trials whose event traces are all zero (2,474 / 51,075 = 4.84 %) are **kept** after the AI verified against the raw NWB that the source `event_detection` block is genuinely all zero there.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
if len(trials) < 2:
    print(f"[pass1 ...] skip {session.ophys_experiment_id}: fewer than 2 kept trials")
    continue
...
except KeyError as exc:                       # raised by get_pupil_data / missing groups
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")
...
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
if centers.size == 0:
    continue
...
if len(neural_trials) < 2:
    raise RuntimeError(f"Session {session.ophys_experiment_id} has fewer than 2 usable trials")
```

iii. Step 3/Step 5: aborted trials are excluded because the animal licked before the change (the SDK also excludes them from rolling performance metrics) and auto-rewarded trials because the free reward biases behaviour; both are explicitly required by the instructions. Key Decision 11: "Three active local files lack eye-tracking acquisition entirely; these sessions will be excluded" because pupil diameter is a required output. Step 10: all-zero-event trials are "valid source-data trials rather than a conversion artifact", verified by a raw-NWB spot check on experiment 792815735 (27 neurons × 388 native frames, raw event sum exactly 0).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is the **event-detection trace** (`processing/ophys/event_detection/data`, stored time × ROI, with its own `timestamps` = ophys frame times). dF/F (`processing/ophys/dff`) is deliberately not used, and neither is the SDK's smoothed `filtered_events`.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. Step 4 discrepancy table: "SDK exposes both `dff_traces` and `events`; paper subset analyses use calcium events … Paper methods explicitly use detected calcium events for neural analyses → Use raw event magnitude traces from `processing/ophys/event_detection/data` as `neural`. Do not use `filtered_events` (visualization-only) and do not recompute dF/F." This tracks the paper methods the AI quoted: "we performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f", and the whitepaper's FastLZero event-detection description.

## 2-b. How is the `neural` data processed?

i. Minimal processing: the (T_native × n_ROI) event matrix is linearly interpolated in time onto each trial's 30 Hz bin-center grid and transposed to (n_neurons, n_timepoints), cast to float32. No smoothing, normalisation, z-scoring, or merging across planes (each plane is its own session). The interpolation is vectorised over neurons using `searchsorted` + broadcasting.

ii.
```python
def linear_resample_matrix(src_time, src_value, dst_time):
    idx_hi = np.searchsorted(src_time, dst_time, side="left")
    idx_hi = np.clip(idx_hi, 1, len(src_time) - 1)
    idx_lo = idx_hi - 1
    t0 = src_time[idx_lo]; t1 = src_time[idx_hi]
    denom = np.where(t1 > t0, t1 - t0, 1.0)
    w = ((dst_time - t0) / denom).astype(np.float32)
    interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
    return interp.T.astype(np.float32, copy=False)
...
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. Step 5 mapping table: "Transpose to ROI x time, then linearly interpolate event magnitudes from native ophys timestamps onto a common 30 Hz trial grid … Use raw event magnitudes, not `filtered_events`." The AI cites the paper's own procedure for building event-triggered responses ("linearly interpolating onto a consistent set of 30hz timestamps … This produces a vector of calcium event magnitudes for each cell"). Step 6 lists the vectorised interpolation as the main speed-up.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No extra neuron-level filtering is applied. The AI relies on the ROI curation already baked into the released NWB (motion-border/duplicate/union/dendrite/dim-ROI removal, `valid_roi`), and verified that every cell listed in `cell_specimen_table` of the included files is `valid_roi == True` (29,168 of 29,168). The per-neuron brain-region vector is built from the length of the cell-specimen table.

ii.
```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

iii. Step 1/Step 10 Check 3: "reference: `CellSpecimens.__init__` keeps `valid_roi == True`; converter: included-session raw NWB files already had all listed cells valid (`29,168` total valid of `29,168` total listed), so event matrices matched converted neuron counts exactly." Step 3 records the whitepaper ROI-exclusion rules and notes they are applied upstream of the released files.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start**: bin centers run from `start_time + DT/2` in steps of DT up to `stop_time`, and every stream (neural, running, pupil, stimulus, outcome) is sampled on that same grid using absolute session time. Metadata records `temporal_alignment_event = "trial start"`, `off_start = 0.0`, `off_end = None` (variable-length trials). The ophys `event_detection` timestamps are the reference clock.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial  = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
pupil_trial   = linear_resample_vector(pupil_time, pupil_diameter, centers)
...
"temporal_alignment_event": "trial start",
"off_start": 0.0,
"off_end": None,
```

iii. Key Decision 7: "Align by absolute ophys time, then cut into trials: For each trial, create bin centers from trial `start_time` to `stop_time` at 30 Hz and sample/interpolate all streams onto that grid." Step 1 notes that the SDK derives all stream timestamps from the same sync file, so absolute times are directly comparable. Step 10 Check 2 verified alignment by reconstructing three trials from raw NWB arrays and comparing with `np.allclose()` (max abs diff ≤ 1.2e-7), and the `--show-processing` plots overlay raw vs resampled traces with the flash boundaries and change time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — everything is rebinned onto a **fixed 30 Hz grid** (`DT = 1/30 s`, `time_bin_size = 33.333 ms`), identical for all trials and sessions. This is a slight downsample for single-plane experiments (native 32.31 ms ≈ 30.95 Hz) and a 3× **upsample** for the multiscope planes (native ≈ 93 ms ≈ 10.7 Hz). Rebinning is done by linear interpolation, not by binning/aggregation, so isolated event magnitudes are split across neighbouring bins and multiscope traces are piecewise-linear between real samples. Resulting T: mean 254, min 211, max 377 bins.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
...
n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
...
"time_bin_size": TIME_BIN_MS,
"sampling_grid_hz": 30.0,
```

iii. Step 4 discrepancy table: "Native sampling rates … 31 Hz single-plane, 11 Hz multiplane … papers/whitepaper describe mixed acquisition rates and also describe 30 Hz interpolation for event-triggered analyses → Resample all streams to a common 30 Hz grid (`33.333... ms` bins). This preserves ophys-based timing while satisfying the decoder requirement that all trials/sessions share a common bin size." Key Decision 6 adds that 30 Hz is also the native rate of the behaviour and eye-tracking streams.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the **stimulus presentation interval tables** in `intervals/*_presentations`, restricted to rows whose `stimulus_block_name` contains `change_detection` (i.e. the active task block): columns `start_time`, `stop_time`, `image_name`, `omitted`, `is_change`, `trials_id`. The trials table's `initial_image_name` / `change_image_name` are read but used only for cross-checking/plotting, not to build the output.

ii.
```python
def get_task_presentations(f: h5py.File) -> pd.DataFrame:
    for name, group in f["intervals"].items():
        if name == "trials": continue
        block_names = decode_str_array(group["stimulus_block_name"][:])
        keep = np.array(["change_detection" in x for x in block_names], dtype=bool)
        ...
        columns = ["start_time", "stop_time", "image_name", "omitted", "is_change",
                   "trials_id", "stimulus_block_name", "active", "duration"]
```
```python
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
```

iii. Step 5 mapping table: "Build per-bin categorical state on 30 Hz grid: actual `image_name` during image flashes; `gray` during gray/omitted periods … Restrict to `stimulus_block_name == change_detection_behavior` when present", referencing the SDK's `get_stimulus_presentations` / `is_change_event`. Key Decision 12 explains the block restriction (ignore natural-movie / spontaneous blocks). The per-flash table is used rather than the trial-level image names because it gives the exact on-screen state flash by flash, including omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global category list is built in pass 1 from all image names seen in the task presentations of all retained sessions, plus an explicit `"gray"` category forced to index 0; the remaining 16 image names are sorted alphabetically (indices 1–16, 17 classes total). For each trial the identity trace is initialised to `gray` everywhere and then, flash by flash, the bins whose centers fall in `[flash.start_time, flash.stop_time)` are set to that flash's image code; omitted flashes (`omitted == True` or `image_name == "omitted"`) are left/set as `gray`. The result is 67.0 % `gray` and ~1.9–2.2 % per image.

ii.
```python
image_names = set(["gray"])
...
image_names.update(x for x in presentations["image_name"].unique() if x != "omitted")
...
image_values = sorted(image_names)
if "gray" in image_values:
    image_values = ["gray"] + [x for x in image_values if x != "gray"]
image_value_to_idx = {name: idx for idx, name in enumerate(image_values)}
```
```python
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    if not mask.any(): continue
    if bool(row.omitted) or str(row.image_name) == "omitted":
        image_identity[mask] = image_value_to_idx["gray"]
    else:
        image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. Key Decision 9: "Encode gray/omission periods explicitly: `image_identity` will include a `gray` category for ISI and omitted-image periods, because the user specifically asks for the image identity during non-gray screen and those periods still occupy trial time." A global (dataset-wide) code map is used so category semantics are identical across sessions; the mapping is exported in `output_values[0]`.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is written on exactly the same `centers` array as the neural matrix: a bin gets an image code iff its center lies inside that flash's `[start_time, stop_time)`. No separate interpolation or shifting; flash times come from the same absolute session clock as the ophys timestamps.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
...
output_trial = np.vstack([image_identity, image_change, running_bin, pupil_bin, outcome_trace])
neural_trials.append(neural_trial.astype(np.float32, copy=False))
output_trials.append(output_trial)
```

iii. Step 5 Decision 7 (single 30 Hz grid per trial for all streams) plus Step 10 Check 2: the image-identity trace for three trials was rebuilt independently from the raw interval tables and matched the converted array exactly; the `--show-processing` plots show the 250 ms / 500 ms flash–gray cadence lining up with the change marker.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the same task stimulus-presentation rows, using the boolean `is_change` column (plus `start_time`/`stop_time`/`trials_id` for placement). The trials table's `change_time`/`go` are not used for the output itself (only in the diagnostic plot).

ii.
```python
columns = ["start_time", "stop_time", "image_name", "omitted", "is_change", "trials_id", ...]
...
out["is_change"] = out["is_change"].fillna(0).astype(bool)
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. Step 1 lists the SDK's `is_change_event`, which "marks image identity changes from successive non-omitted images", as the reference implementation; Step 5 maps it to a "Binary per-bin trace: 1 during change-image flash interval, else 0", with the trial `change_time` kept as a cross-check. Because a catch trial's sham change repeats the same image, `is_change` is False there, so catch trials get an all-zero change trace.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector of length `n_bins` is filled with 1 for the bins whose centers lie inside the change flash's `[start_time, stop_time)` window — i.e. the 250 ms image presentation only (~7–8 bins at 30 Hz), not the following grey period. Across the dataset this gives 2.5 % positive bins.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    if not mask.any(): continue
    ...
    if bool(row.is_change):
        image_change[mask] = 1
```

iii. Step 5 mapping table: "Binary per-bin trace: 1 during change-image flash interval, else 0 … Change and pre-change flashes are never omitted." This implements the instruction "Have value of 1 right after a change in image identity, otherwise 0" as "during the changed flash", keeping the positive class tied to the physical stimulus event.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — the variable is natively binary (`no_change` = 0, `change` = 1), produced directly from the boolean `is_change` flag and the flash window. `output_values[1] = ["no_change", "change"]`.

ii.
```python
"output_values": [
    image_values,
    ["no_change", "change"],
    RUNNING_BIN_NAMES,
    PUPIL_BIN_NAMES,
    OUTCOME_NAMES,
],
```

iii. The instruction specifies image change as a binary variable; the AI notes in Step 12 that the positive class is sparse (2.5 % of bins) but that "balanced accuracy accounts for class imbalance", so no rebalancing or widening of the window was applied.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same grid and same masking operation as image identity — the change flag occupies the bins of the change flash on the shared trial `centers` array, so it is sample-for-sample aligned with the neural matrix.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
if bool(row.is_change):
    image_change[mask] = 1
```

iii. Same justification as 3-c: one 30 Hz grid per trial built from absolute times, verified by the raw-data `np.allclose()` spot checks in Step 10 Check 2 and visually in `processing_775614751.png` / `processing_788490510.png` (change indicator lines up with the change flash and with `change_time`).

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `processing/running/speed` (`data` + `timestamps`) — the SDK's filtered running speed in cm/s (the unfiltered `speed_unfiltered` series is not used).

ii.
```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. Step 5 mapping table: "`processing/running/speed` (`data`, `timestamps`) → `output[running_speed_bin]` … Use filtered running speed in cm/s", referencing `RunningSpeed.from_stimulus_file` / `_get_running_speed_df` in the SDK, which computes speed from the wheel encoder on stimulus timestamps with no monitor-delay offset.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation (`np.interp`) of the raw speed trace onto the trial's 30 Hz bin centers, then discretisation with globally computed quantile edges. No smoothing, rectification or clipping of negative speeds (negative values occur and fall in the lowest bins).

ii.
```python
def linear_resample_vector(src_time, src_value, dst_time):
    ...
    return np.interp(dst_time, src_time, src_value).astype(np.float32, copy=False)
...
running_trial = linear_resample_vector(running_time, running_speed, centers)
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. Step 5: "Interpolate filtered running speed onto 30 Hz trial grid; discretize globally across included data into 5 equal-frequency bins." Interpolation is used because the running stream is sampled at 60 Hz on its own clock and must be placed on the common trial grid; the AI verified alignment by independently reconstructing the pre-discretisation running trace from raw NWB arrays (Step 10 Check 2).

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins with edges computed **globally** in pass 1, over the resampled running values of every kept trial of every kept session (finite values only). `np.quantile` at [0, 0.2, …, 1.0]; values are clipped to the outer edges and assigned with `searchsorted` on the interior edges; duplicate edges are nudged apart with epsilons. Achieved distribution: 0.200 / 0.200 / 0.200 / 0.200 / 0.200.

ii.
```python
def compute_quantile_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    probs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(values, probs)
    if np.unique(edges).size < edges.size:
        eps = np.finfo(np.float64).eps
        edges = np.maximum.accumulate(edges + np.arange(edges.size) * eps)
    return edges

def digitize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, edges[0], edges[-1])
    bins = np.searchsorted(edges[1:-1], clipped, side="right")
    return bins.astype(np.int64, copy=False)
...
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
```

iii. Key Decision 10: "Discretize running and pupil globally across the included dataset: Five equal-percentile bins will be computed from all finite samples across all included sessions/trials, not per session, so class semantics are consistent dataset-wide." This follows the instruction "discretized into five equal percentile bins"; the edges are exported in `metadata['running_bin_edges']`.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated directly onto the same `centers` grid used for the neural matrix, so index *t* of the running-bin row corresponds to index *t* of the neural matrix.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial  = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. Step 1 notes that running-speed timestamps come from the same sync clock as the ophys frames (with no monitor-delay offset), so absolute-time interpolation is valid; the diagnostic plot overlays the raw running trace and the resampled trace over the trial window to make misalignment visible.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. From `acquisition/EyeTracking/pupil_tracking` — the `width` and `height` of the fitted pupil ellipse plus `eye_tracking/timestamps`. Diameter is defined as `2 × max(width, height)`. Blink frames are handled implicitly: in these NWBs the pupil width/height are already NaN exactly on `likely_blink` frames (verified: NaN mask == `likely_blink`, ~9 % of frames), and those NaNs are filled by time interpolation. Sessions with no `EyeTracking` group at all are dropped.

ii.
```python
def get_pupil_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    if "EyeTracking" not in f["acquisition"]:
        raise KeyError("Missing EyeTracking acquisition")
    eye_group = f["acquisition"]["EyeTracking"]
    timestamps = np.asarray(eye_group["eye_tracking"]["timestamps"][:], dtype=np.float64)
    width = np.asarray(eye_group["pupil_tracking"]["width"][:], dtype=np.float64)
    height = np.asarray(eye_group["pupil_tracking"]["height"][:], dtype=np.float64)
    diameter = 2.0 * np.maximum(width, height)
    diameter = fill_nan_by_time(timestamps, diameter)
    return timestamps, diameter
```

iii. Step 5 mapping table: "Compute pupil diameter as `2 * max(width, height)` after blink filtering; interpolate onto 30 Hz grid", referencing `process_eye_tracking_data` / `filter_on_blinks` and the whitepaper eye-tracking description. This mirrors the SDK's own convention in `compute_circular_area`, whose docstring reads "Calculate the area of the pupil as a circle using the max of the height/width as radius" — i.e. `max(width, height)` is the radius, so the diameter is twice it. Key Decision 11 covers dropping sessions without eye tracking and interpolating blink NaNs.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) diameter = 2·max(width, height); (2) NaN (blink) samples filled by linear interpolation in time on the native eye-tracking clock; (3) linear interpolation onto the trial's 30 Hz bin centers; (4) discretisation with global quantile edges (same machinery as running speed). If a session's pupil trace were entirely NaN, `fill_nan_by_time` raises.

ii.
```python
def fill_nan_by_time(time_axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=True)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        raise ValueError("All values are NaN")
    if finite.all():
        return values
    values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
    return values
...
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. Step 5 / Key Decision 11: blink-related missingness is "modest" and can be "filled by time interpolation before discretization", which avoids injecting an artificial category and keeps a continuous signal to percentile-bin. Filling before resampling prevents blink artifacts (or NaNs) from leaking into neighbouring bins.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical scheme to running speed: five global equal-percentile bins computed in pass 1 over all finite resampled pupil samples of all kept trials, applied with clip + `searchsorted`. Achieved distribution: 0.200 per bin. Edges exported in `metadata['pupil_bin_edges']`.

ii.
```python
pupil_all = np.concatenate(pupil_values).astype(np.float64, copy=False)
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
...
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. Key Decision 10 (same as running speed): global rather than per-session percentiles so that a given bin label means the same thing dataset-wide, as required by "discretized into five equal percentile bins".

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto the same per-trial 30 Hz `centers` array as the neural data, so it is index-aligned with the neural matrix.

ii.
```python
pupil_time, pupil_diameter = get_pupil_data(f)
...
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. Same rationale as running speed — the eye-tracking frames carry sync-derived timestamps on the same clock as the ophys frames, so absolute-time interpolation aligns the streams; verified by the independent raw-data reconstruction in Step 10 Check 2 and by the raw-vs-resampled pupil panel in the processing plots.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. From the four mutually exclusive boolean columns of the NWB trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, checked in that order and mapped to 0–3. A trial with none of the four set raises an error rather than being silently labelled.

ii.
```python
def trial_outcome_index(trial_row: pd.Series) -> int:
    if bool(trial_row["hit"]): return 0
    if bool(trial_row["miss"]): return 1
    if bool(trial_row["false_alarm"]): return 2
    if bool(trial_row["correct_reject"]): return 3
    raise ValueError("Trial has no valid outcome label")
```

iii. Step 1 notes that `Trial._get_trial_data` "defines `go`, `catch`, `aborted`, `auto_rewarded`, hit/miss/FA/CR logic", and that auto-rewarded trials clear these labels — which is why the AI filters them out first. Step 5 maps the flags to `output[trial_outcome]`; Step 10 Check 4 recomputed the outcome fractions from the raw tables ([0.307, 0.568, 0.018, 0.107]) and matched the converted file exactly.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The single per-trial integer code is broadcast to a constant row over all bins of the trial so that the output block has a uniform `(5, n_timepoints)` shape. Class names are fixed in `OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]`.

ii.
```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)
output_trial = np.vstack([image_identity, image_change, running_bin, pupil_bin, outcome_trace])
```

iii. Key Decision 8: "Represent all outputs as time-varying traces … `trial_outcome` will be repeated across bins within a trial as a constant categorical trace to keep one consistent `(n_output, T)` format", which follows the target-format guidance to prefer time-varying outputs where possible.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases: missing `EyeTracking` group → the whole session is skipped (`KeyError` caught in pass 1; 3 sessions); blink/NaN pupil samples → filled by linear interpolation in time; sessions with < 2 kept trials → skipped; degenerate trial windows (non-finite or non-positive duration) → skipped; duplicate quantile edges (degenerate distributions) → nudged apart by epsilons; string datasets stored as bytes/objects → decoded defensively; missing optional columns in interval tables → simply not read; `omitted`/`is_change` NaNs → filled with False. All-zero event trials (4.84 %) are knowingly retained after raw-data verification. Not handled: NaNs in running speed (none exist in the data), a failure inside pass 2 is uncaught and would abort the run, and bin centers outside the ophys timestamp range would be linearly extrapolated rather than clipped (checked: no kept trial falls outside the ophys range).

ii.
```python
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
...
except KeyError as exc:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")
...
values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
...
if not np.isfinite(start_time) or not np.isfinite(stop_time) or stop_time <= start_time:
    return np.asarray([], dtype=np.float64)
...
out["omitted"] = out["omitted"].fillna(0).astype(bool)
out["is_change"] = out["is_change"].fillna(0).astype(bool)
```

iii. Step 5 Key Decision 11 and Step 10 Check 5 ("Edge cases"): sessions missing eye tracking are excluded up front "because pupil output is required"; blink NaNs are interpolated rather than binned as a separate class; "Trial binning uses centers strictly within `[start_time, stop_time)`, avoiding off-by-one inclusion past trial end"; "All-zero event trials are retained because they are present in the source data and still have valid behavioral/stimulus labels", a claim the AI backed with a direct raw-NWB inspection of experiment 792815735.

## 9-a. What are the most time-consuming steps of the code?

i. The dominant costs are I/O plus the per-trial resampling: reading the full (T_native × n_ROI) `event_detection` matrix and the running/eye-tracking arrays out of each NWB (done once per pass, i.e. twice per file), and the per-trial fancy-index + broadcast interpolation in `linear_resample_matrix`, which allocates an (n_bins × n_neurons) array per trial. Measured in the full run: ~0.4–0.7 s/session in pass 1 and ~0.7–2.0 s/session in pass 2, 395.6 s total for 202 + 199 session reads. Building the diagnostic plots is expensive but only runs with `--show-processing`.

ii.
```python
# pass 1 — reads trials, presentations, running and eye tracking for every session
running_time, running_speed = get_running_data(f)
pupil_time, pupil_diameter = get_pupil_data(f)
for trial in trials.itertuples(index=False):
    running_trial = linear_resample_vector(running_time, running_speed, centers)
    pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
# pass 2 — reads the same file again, plus the full event matrix
ophys_time, events = get_neural_data(f)
for trial_idx, trial in trials.iterrows():
    neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. Step 6: "Full conversion may still be I/O-heavy because each NWB event matrix must be read from disk. Global binning requires a first pass over sessions, so conversion reads each file twice." Step 7 gives the measured per-session timings and the ~7.5 min projection, which the full run (6.6 min) matched.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Already vectorised: the neural interpolation across neurons (`linear_resample_matrix` replaces a per-neuron `np.interp` loop). Remaining Python-level loops that could be vectorised: the per-trial loops in both `collect_global_statistics` and `convert_session` (all trials of a session could be resampled in one `searchsorted`/gather over a concatenated grid); the inner per-stimulus-flash loop that fills `image_identity` / `image_change` (could be a single `searchsorted` of `centers` into the flash boundary array); `decode_str_array`, which decodes every string element in a Python loop (`np.char.decode` / `astype(str)` would be vectorised); and pass 1's per-trial running/pupil resampling, which for quantile purposes could be done once per session on the whole trace.

ii.
```python
for trial_idx, trial in trials.iterrows():          # per-trial loop (pass 2)
    ...
    for row in trial_presentations.itertuples(index=False):   # per-flash loop
        mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
...
def decode_str_array(values: np.ndarray) -> np.ndarray:
    out = []
    for value in values:                            # per-element Python loop
        ...
```

iii. Step 6 "Code speedups added": "Vectorized linear interpolation for neural event matrices using `searchsorted` + broadcasting rather than per-neuron `np.interp`." The AI judged the remaining loops acceptable because the run finished in ~6.6 min, well inside the 15-minute budget set by the instructions.

## 9-c. What processing does the code repeat multiple times?

i. Every retained NWB is opened and parsed **twice**: pass 1 reads the trials table, the stimulus presentation tables, running speed and eye tracking, builds the per-trial bin grids and resamples running/pupil onto them purely to accumulate values for the global quantile edges; pass 2 re-reads and re-does all of that work (plus the neural matrix). So trial filtering, bin-grid construction, and running/pupil resampling each happen twice for every trial in the dataset. `build_trial_bins` is also called twice per trial, and `get_task_presentations` re-scans all `intervals` groups in both passes.

ii.
```python
# pass 1
trials = get_trial_table(f); presentations = get_task_presentations(f)
running_time, running_speed = get_running_data(f); pupil_time, pupil_diameter = get_pupil_data(f)
for trial in trials.itertuples(index=False):
    centers = build_trial_bins(float(trial.start_time), float(trial.stop_time))
    running_values.append(linear_resample_vector(running_time, running_speed, centers))
    pupil_values.append(linear_resample_vector(pupil_time, pupil_diameter, centers))
# pass 2 (same work repeated)
trials = get_trial_table(f); presentations = get_task_presentations(f)
running_time, running_speed = get_running_data(f); pupil_time, pupil_diameter = get_pupil_data(f)
for trial_idx, trial in trials.iterrows():
    centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
    running_trial = linear_resample_vector(running_time, running_speed, centers)
    pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. Step 6 documents the trade-off explicitly: "Uses a two-pass conversion: 1. pass 1 computes global running-speed and pupil-diameter percentile edges and filters out unusable sessions; 2. pass 2 converts trials into session/trial matrices", and "Session-level streaming design avoids storing continuous raw traces for the whole dataset in memory." The two passes are needed because the quantile edges must be global, but the pass-1 results (small float32 traces) are thrown away rather than cached.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Pass 1's resampled running/pupil per-trial traces are used only to compute ten quantile edges and are then discarded — the same edges could have been obtained from the raw (unresampled) session traces or a subsample. (2) Pass 1 also builds per-trial bin grids and parses the presentation tables for image names that pass 2 recomputes. (3) Upsampling the 34 multiscope planes from ~10.7 Hz to 30 Hz roughly triples their stored timepoints without adding information, inflating the pickle (8.1 GB) and the decoder's PCA cost. (4) `get_pupil_data` reads and combines `height` as well as `width`, and `get_task_presentations` reads `active` and `duration` columns that are never used. (5) `input` is materialised as a `(0, T)` array per trial even though there are no decoder inputs. (6) `SessionMeta` carries `ophys_session_id`, `behavior_session_id`, `session_type`, `experience_level` and `project_code` that are only echoed into metadata (`project_code` is never used at all — no project filter is applied). (7) An unused `import math`. None of this was flagged by the AI except the double file read.

ii.
```python
running_values.append(running_trial)   # pass 1: kept only to compute 10 quantile edges
pupil_values.append(pupil_trial)
...
input_trials = [np.zeros((0, trial.shape[1]), dtype=np.float32) for trial in neural_trials]
...
import math   # unused
project_code: str      # stored, never used for filtering
```

iii. The AI documented only the double-read cost ("Global binning requires a first pass over sessions, so conversion reads each file twice", Step 6) and accepted it as the price of dataset-wide percentile bins; it justified the 30 Hz grid for all sessions (including the 11 Hz multiscope planes) as necessary for a single common `time_bin_size` (Key Decision 6). The remaining items are not discussed in CONVERSION_NOTES.md.
