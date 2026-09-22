# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did **not** use the AllenSDK high-level loader. It reported that the local AllenSDK/pynwb stack could not instantiate these NWB files (`external_resources`/version mismatch), so it reads the released NWB files directly with `h5py` and uses the released CSV manifest for metadata. Concretely:
- Session/experiment inventory comes from `data/visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv`.
- The table is intersected with the NWB files actually present on disk (`behavior_ophys_experiment_<id>.nwb`, 284 files), then filtered to `passive == False` (202 active experiments) and sorted by `ophys_experiment_id`.
- No `project_code` filter is applied, so both `VisualBehavior` (168 active) and `VisualBehaviorMultiscope` (34 active) experiments are included.
- Per-file, the processed NWB groups are read directly: `intervals/trials`, `intervals/*_presentations`, `processing/ophys/event_detection`, `processing/running/speed`, `acquisition/EyeTracking`, `processing/ophys/image_segmentation/cell_specimen_table`.
- A two-pass design is used: pass 1 opens every file to build global running/pupil quantile edges and the global image vocabulary and to drop unusable sessions; pass 2 re-opens every kept file and builds the trial matrices.

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
all_sessions = get_local_session_metadata(data_root)
print(f"Found {len(all_sessions)} local active experiment files")
...
print("Pass 1: collecting global running/pupil statistics")
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
print("Pass 2: converting sessions")
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

```python
with h5py.File(session.filepath, "r") as f:
    trials = get_trial_table(f)
    presentations = get_task_presentations(f)
    ophys_time, events = get_neural_data(f)
    running_time, running_speed = get_running_data(f)
    pupil_time, pupil_diameter = get_pupil_data(f)
```

iii. From CONVERSION_NOTES.md Step 4: *"Local environment cannot instantiate `NWBFile` for these NWBs due `external_resources`/version mismatch … Read NWB files directly with `h5py` and mirror the AllenSDK/whitepaper semantics from the processed NWB contents rather than relying on broken high-level loading in this environment."* Step 10 Check 3 adds: *"same processed sources are used; only the loader mechanism differs due environment incompatibility."* Passive sessions are excluded because *"Passive sessions are not task performance and make outcome labels degenerate"* (Step 4/Step 5 Key Decision 1).

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` strings taken from `ophys_experiment_table.csv`, collected over the sessions that survive pass-1 QC, sorted lexicographically. `subject_idx[session]` is the index of that session's `mouse_id` in the subject list. Result: 38 subjects.

ii.
```python
mouse_id=str(row.mouse_id),      # SessionMeta field, from ophys_experiment_table.csv
...
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. CONVERSION_NOTES.md Step 5 variable mapping: *"`mouse_id` from `ophys_experiment_table.csv` → `subjects`, `subject_idx`; Unique string list + per-session index."* `mouse_id` is the canonical animal identifier in the Allen metadata table, so no further derivation was needed.

## 1-c. How are the data split into sessions?

i. **Each `ophys_experiment_id` (i.e. each imaging plane/NWB file) is treated as one converted "session."** The AI explicitly decided *not* to group planes recorded simultaneously under the same `ophys_session_id`/`behavior_session_id`. Sessions are ordered by `ophys_experiment_id`, not by acquisition date. Because 34 of the retained experiments are `VisualBehaviorMultiscope` planes from 6 real recording sessions of one mouse (457841), the same behavioral/stimulus trial data is emitted 5–8 times as separate "sessions" with different neuron sets (visible in the verification log as repeated per-session trial counts `209, 209, 209, …`, `287 ×7`, `309 ×7`, and as "Subject 457841: 34 sessions"). Final count: 199 sessions.

ii.
```python
sessions.append(
    SessionMeta(
        ophys_experiment_id=int(row.ophys_experiment_id),
        ophys_session_id=int(row.ophys_session_id),
        behavior_session_id=int(row.behavior_session_id),
        ...
    )
)
```

```python
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(session=session, ...)
    neural_all.append(neural_trials)
    ...
    brain_region_idx_all.append(brain_region_idx)
```

```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 2: *"Treat each `ophys_experiment_id` file as one converted session: This matches the AllenSDK object granularity (`BehaviorOphysExperiment`) and yields a single imaging plane / neuron set / brain region per session."* Step 10 Check 5 acknowledges the consequence explicitly: *"Multiple experiment files can share the same `behavior_session_id` or `ophys_session_id`; the conversion intentionally treats each `ophys_experiment_id` plane as a separate session because that is the AllenSDK experiment granularity and each file has its own neuron set."*

## 1-d. How are the data split into trials?

i. Trials come from the processed NWB trials table (`intervals/trials`), the same table the AllenSDK exposes as `dataset.trials`. Kept trials satisfy `(go | catch) & ~aborted & ~auto_rewarded`. Each trial spans the full `start_time → stop_time` window (variable length, mean ≈ 8.5 s / ≈ 256 bins at 30 Hz), resampled onto a regular 30 Hz grid of bin centers starting half a bin after `start_time`. Trials whose window produces zero bins are dropped.

ii.
```python
def build_trial_bins(start_time: float, stop_time: float) -> np.ndarray:
    if not np.isfinite(start_time) or not np.isfinite(stop_time) or stop_time <= start_time:
        return np.asarray([], dtype=np.float64)
    n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
    centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
    valid = centers < (stop_time + 1e-9)
    return centers[valid]
```

```python
trials = get_trial_table(f)
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
trials = trials.sort_values("id").reset_index(drop=True)
...
for trial_idx, trial in trials.iterrows():
    centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
    if centers.size == 0:
        continue
```

iii. CONVERSION_NOTES.md Step 5 Key Decisions 3–4: *"Segment trials using the processed NWB `trials` table: This mirrors the Allen reference processing without re-deriving trial logic from lower-level files"* and *"Keep only contingent trials: Include GO and CATCH trials; exclude `aborted` and `auto_rewarded` exactly as required and consistent with reference definitions."* Step 1 notes cite `allensdk/brain_observatory/behavior/trial_masks.py::contingent_trials` as the SDK's own definition of GO+CATCH, and `Trial._get_trial_data` as the source of the mutually exclusive outcome flags.

## 1-e. How are trials filtered based on quality controls?

i. Filters applied, in order:
- Session level (pass 1): drop sessions with `< 2` contingent trials; drop sessions with no task (`change_detection`) stimulus presentation block; drop sessions whose NWB has no `acquisition/EyeTracking` group (3 sessions: 795953296, 806456687, 833631914).
- Trial level: only `(go | catch) & ~aborted & ~auto_rewarded`; trials producing an empty 30 Hz bin grid are skipped.
- Session level (pass 2): a session that ends up with `< 2` usable trials raises `RuntimeError`.
- **Not** filtered: 2,474/51,075 (4.84%) of kept trials have an all-zero event trace; the AI verified against the raw NWB that these are genuinely zero in the source `event_detection` matrix and deliberately retained them.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
if len(trials) < 2:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: fewer than 2 kept trials")
    continue

presentations = get_task_presentations(f)
if presentations.empty:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: no task presentations")
    continue
...
except KeyError as exc:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")
```

```python
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
```

```python
if len(neural_trials) < 2:
    raise RuntimeError(f"Session {session.ophys_experiment_id} has fewer than 2 usable trials")
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 11: *"Require pupil availability at session level: Three active local files lack eye-tracking acquisition entirely; these sessions will be excluded."* Step 10: *"All-zero event trials are retained because they are present in the source data and still have valid behavioral/stimulus labels"*, backed by a raw spot-check (*"session index 3, experiment 792815735, kept trial index 2, 27 neurons, 388 native frames, raw event sum 0.0"*). The `< 2` trial rule exists because the decoder needs at least two trials per session to evaluate.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is the **event-detection trace** (`processing/ophys/event_detection/data`, stored time × ROI, with `…/timestamps` giving ophys frame times) — i.e. the L0/FastLZero-detected calcium event magnitudes, **not** dF/F and **not** `filtered_events`.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. CONVERSION_NOTES.md Step 4: *"Paper methods explicitly use detected calcium events for neural analyses → Use raw event magnitude traces from `processing/ophys/event_detection/data` as `neural`. Do not use `filtered_events` (visualization-only) and do not recompute dF/F."* Step 3 cites `methods.txt:179`/`:208` (*"Paper analyses frequently use discrete calcium events rather than raw dF/F"*) and the whitepaper's FastLZero event-detection description. The paper methods text the agent extracted reads: *"This process produces, for each cell, a set of calcium events each with a time and magnitude … We compute the behavioral event triggered response by isolating the calcium events around the triggering behavioral event, then linearly interpolating onto a consistent set of 30hz timestamps."*

## 2-b. How is the `neural` data processed?

i. Minimal processing: the (time × ROI) event matrix is linearly interpolated from native ophys timestamps onto the trial's 30 Hz bin-center grid and transposed to (n_neurons, n_timepoints), cast to `float32`. No smoothing, normalisation, z-scoring, baseline subtraction, or cross-plane merging is performed (planes are separate sessions, see 1-c). The interpolation is vectorised across all neurons using `searchsorted` + broadcasting rather than per-neuron `np.interp`.

ii.
```python
def linear_resample_matrix(src_time, src_value, dst_time) -> np.ndarray:
    """Resample a time x features matrix onto dst_time."""
    if dst_time.size == 0:
        return np.zeros((src_value.shape[1], 0), dtype=np.float32)

    idx_hi = np.searchsorted(src_time, dst_time, side="left")
    idx_hi = np.clip(idx_hi, 1, len(src_time) - 1)
    idx_lo = idx_hi - 1

    t0 = src_time[idx_lo]
    t1 = src_time[idx_hi]
    denom = np.where(t1 > t0, t1 - t0, 1.0)
    w = ((dst_time - t0) / denom).astype(np.float32)
    interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
    return interp.T.astype(np.float32, copy=False)
```

```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
...
neural_trials.append(neural_trial.astype(np.float32, copy=False))
```

iii. CONVERSION_NOTES.md Step 5 mapping: *"Transpose to ROI x time, then linearly interpolate event magnitudes from native ophys timestamps onto a common 30 Hz trial grid … Use raw event magnitudes, not `filtered_events`."* Step 6: *"Vectorized linear interpolation for neural event matrices using `searchsorted` + broadcasting rather than per-neuron `np.interp`."* The linear-interpolation-to-30 Hz step is taken verbatim from the paper's event-triggered-response methods.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level quality filtering is applied. All ROIs in the NWB `cell_specimen_table` (equivalently, all columns of the event matrix) are kept. The AI checked that the released files already contain only valid ROIs: *"included-session raw NWB files already had all listed cells valid (29,168 total valid of 29,168 total listed)"*. (I independently confirmed on 60 files: 1,787/1,787 ROIs have `valid_roi == True`, and the event matrix column count always equals the cell-table row count.) Total: 29,168 neurons.

ii.
```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```
(No `valid_roi` mask is applied anywhere in `convert_data.py`.)

iii. CONVERSION_NOTES.md Step 10 Check 3: *"reference: `CellSpecimens.__init__` keeps `valid_roi == True`; converter: included-session raw NWB files already had all listed cells valid … so event matrices matched converted neuron counts exactly."* Step 3 documents that the Allen pipeline already removed motion-border, duplicate, union, dendritic and too-small/dim ROIs upstream of the public release, so no further curation was deemed necessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start** (`trials.start_time`) on the absolute ophys clock. For each trial, bin centers are generated at `start_time + (k + 0.5)·(1/30) s` for all centers `< stop_time`; the event trace is then linearly interpolated at those absolute times. Because every stream (neural, running, pupil, stimulus, outcome) is sampled on the *same* `centers` vector, all streams are aligned by construction. Metadata records `temporal_alignment_event = "trial start"`, `off_start = 0.0`, `off_end = None`.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
if centers.size == 0:
    continue

neural_trial = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
pupil_trial   = linear_resample_vector(pupil_time, pupil_diameter, centers)
...
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
```

```python
"temporal_alignment_event": "trial start",
"off_start": 0.0,
"off_end": None,
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 7: *"Align by absolute ophys time, then cut into trials: For each trial, create bin centers from trial `start_time` to `stop_time` at 30 Hz and sample/interpolate all streams onto that grid."* Step 10 Check 2 reports raw-NWB `np.allclose()` spot checks on three trials (max abs diff 2.98e-08, 0.0, 1.19e-07 for neural; exact match for all output rows), plus visual verification in `processing_775614751.png` / `processing_788490510.png`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Yes, rebinning is applied.** All streams are resampled onto a common 30 Hz grid, `time_bin_size = 1000/30 = 33.33 ms`, identical for every trial and every session. Native acquisition is 31 Hz for single-plane and ~11 Hz per plane for multiscope, so this is a mild downsample for single-plane data and an ~3× *upsample* (by linear interpolation of sparse event magnitudes) for the multiscope planes. Resampling is by linear interpolation, not by integrating/summing into bins.

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

iii. CONVERSION_NOTES.md Step 5 Key Decision 6: *"Use a common 30 Hz trial grid: Native acquisition rates vary across rigs (31 Hz single-plane, 11 Hz multiplane). Resampling all streams to 30 Hz gives one shared bin size while remaining close to behavior/eye-tracking rate and consistent with paper event-triggered interpolation onto 30 Hz timestamps."* Step 4 adds: *"papers/whitepaper describe mixed acquisition rates and also describe 30 Hz interpolation for event-triggered analyses."* This is also required by the target-format rule *"Time bins should be the same size for all trials and sessions"*, given that the AI includes both rig types.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is built from the **stimulus presentation interval tables** (`intervals/*_presentations`), restricted to rows whose `stimulus_block_name` contains `change_detection`. The columns used are `image_name`, `omitted`, `start_time`, `stop_time` and `trials_id`. The trials-table fields `initial_image_name` / `change_image_name` are read but **not** used to build the output (they are only read into the trial dataframe).

ii.
```python
def get_task_presentations(f: h5py.File) -> pd.DataFrame:
    rows = []
    for name, group in f["intervals"].items():
        if name == "trials":
            continue
        if "stimulus_block_name" not in group or "start_time" not in group:
            continue
        block_names = decode_str_array(group["stimulus_block_name"][:])
        keep = np.array(["change_detection" in x for x in block_names], dtype=bool)
        ...
        columns = ["start_time", "stop_time", "image_name", "omitted", "is_change",
                   "trials_id", "stimulus_block_name", "active", "duration"]
```

iii. CONVERSION_NOTES.md Step 5 mapping: *"`intervals/*_presentations` task image block … → `output[image_identity]`; Build per-bin categorical state on 30 Hz grid: actual `image_name` during image flashes; `gray` during gray/omitted periods"*, and Key Decision 12: *"Restrict stimulus rows to the task block … ignoring natural-movie or spontaneous blocks stored elsewhere in NWB."* Step 1 cites the SDK's `get_stimulus_presentations` / `is_change_event` as the reference implementation being mirrored.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A **17-level** categorical trace is produced per trial: the 16 natural images seen anywhere in the dataset plus an explicit `"gray"` class (forced to index 0). Every bin is initialised to `gray`; for each stimulus presentation assigned to the trial, bins whose center falls in `[presentation.start_time, presentation.stop_time)` get that presentation's image code; omitted flashes stay `gray`. The vocabulary is built globally in pass 1 across all sessions and sorted, so codes are consistent dataset-wide. Result: `gray` occupies 67.0% of bins (the expected 500/750 ms grey duty cycle), each image ≈ 1.9–2.2%.

ii.
```python
image_values = sorted(image_names)
if "gray" in image_values:
    image_values = ["gray"] + [x for x in image_values if x != "gray"]
image_value_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

```python
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
...
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    if not mask.any():
        continue
    if bool(row.omitted) or str(row.image_name) == "omitted":
        image_identity[mask] = image_value_to_idx["gray"]
    else:
        image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 9: *"Encode gray/omission periods explicitly: `image_identity` will include a `gray` category for ISI and omitted-image periods, because the user specifically asks for the image identity during non-gray screen and those periods still occupy trial time."* Key Decision 10 (global vocabulary) is applied here too so class semantics are dataset-wide.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed directly on the same `centers` array used to resample the neural data, using the presentations' absolute `start_time`/`stop_time`. No separate alignment step exists, so the image trace and the neural matrix share indices by construction. The AI verified in Step 10 that the reconstructed image trace matches the converted array exactly for three trials, and visually in the `--show-processing` plots (flashes drawn as shaded spans over the neural traces).

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
...
image_identity[mask] = image_value_to_idx[str(row.image_name)]
```
(with `neural_trial = linear_resample_matrix(ophys_time, events, centers)` using the same `centers`)

iii. Step 10 Check 2: *"`image_identity`, `image_change`, `running_speed_bin`, `pupil_diameter_bin`, `trial_outcome` all matched exactly"* for the three independently reconstructed trials. Step 7: plots *"show image flashes alternating with gray periods at the expected 250 ms / 500 ms cadence"*.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the `is_change` column of the same task stimulus presentation table (plus that presentation's `start_time`/`stop_time`). The trials-table `change_time` is used only as a cross-check/plot annotation, not to build the output.

ii.
```python
columns = ["start_time", "stop_time", "image_name", "omitted", "is_change", "trials_id", ...]
df = read_interval_table(group, columns)
...
out["is_change"] = out["is_change"].fillna(0).astype(bool)
```

iii. CONVERSION_NOTES.md Step 5 mapping: *"Same stimulus-presentation rows → `output[image_change]`; Binary per-bin trace: 1 during change-image flash interval, else 0 … `is_change_event`; trial `change_time` for cross-check. Change and pre-change flashes are never omitted."*

## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary per-bin trace, initialised to 0, set to 1 for the bins covered by the flash whose `is_change` is True (i.e. one 250 ms flash per go trial). Catch trials carry no `is_change` flash, so their trace is all zeros. Resulting distribution: 2.5% ones / 97.5% zeros.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    if not mask.any():
        continue
    ...
    if bool(row.is_change):
        image_change[mask] = 1
```

iii. Per the mapping table, the binary indicator marks *"1 during change-image flash interval, else 0"*. The instruction asks for a variable that is *"1 right after a change in image identity, otherwise 0"*, and the AI took the change flash itself (250 ms) as that interval. Step 12 notes the consequence: *"image change has sparse positives (2.5% of bins) but balanced accuracy accounts for class imbalance."*

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is required or applied — the variable is natively binary (`{0, 1}`), labelled `["no_change", "change"]` in `output_values`. The only implicit "thresholding" is the choice of temporal extent of the `1` state (the 250 ms change flash).

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

iii. Not discussed as a thresholding decision in CONVERSION_NOTES.md, because the instruction already specifies the variable as binary ("Image change, binary variable").

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity — the mask is evaluated on the same `centers` grid used for the neural resampling, so it is aligned by construction. The change flash is verified against the trials-table `change_time` in the diagnostic plots (red dashed line at `change_time - start_time`).

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
if bool(row.is_change):
    image_change[mask] = 1
...
if np.isfinite(trial_row["change_time"]):
    ax0.axvline(float(trial_row["change_time"]) - trial_start, color="red", linestyle="--", linewidth=1.5)
```

iii. Step 7: *"change indicator aligned to the change flash"*; Step 10 Check 2 confirms exact match against independent raw-NWB reconstruction.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed` (`data` + `timestamps`) — the SDK's filtered running speed in cm/s. The unfiltered variant (`processing/running/speed_unfiltered`) and the raw `dx` are not used.

ii.
```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. CONVERSION_NOTES.md Step 5 mapping: *"`processing/running/speed` (`data`, `timestamps`) → `output[running_speed_bin]` … Use filtered running speed in cm/s"*, referencing `RunningSpeed.from_stimulus_file` from Step 1 (which notes running timestamps deliberately carry **no** monitor-delay correction).

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation (`np.interp`, which clamps rather than extrapolates outside the recorded range) onto the trial's 30 Hz `centers`, cast to `float32`, then discretisation into 5 global quantile bins (see 5-c). No smoothing, clipping of negative speeds, or per-session normalisation.

ii.
```python
def linear_resample_vector(src_time, src_value, dst_time) -> np.ndarray:
    if dst_time.size == 0:
        return np.asarray([], dtype=np.float32)
    if src_time.size == 0:
        raise ValueError("Cannot resample from an empty source time series")
    return np.interp(dst_time, src_time, src_value).astype(np.float32, copy=False)
```

```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. Step 5 mapping: *"Interpolate filtered running speed onto 30 Hz trial grid; discretize globally across included data into 5 equal-frequency bins."*

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five **equal-frequency (quantile) bins** with edges computed globally in pass 1 from all finite resampled running samples over all kept trials of all kept sessions (not per session). Edges found: `[-24.04, -0.00407, 0.303, 15.77, 32.2…, max]`. Values are clipped to `[edge0, edge_last]` and assigned with `searchsorted(..., side="right")` on the interior edges. Degenerate (duplicate) edges are nudged by machine epsilon to keep them strictly increasing. Achieved distribution is exactly 0.200 per class.

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
```

```python
running_all = np.concatenate(running_values).astype(np.float64, copy=False)
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 10: *"Discretize running and pupil globally across the included dataset: Five equal-percentile bins will be computed from all finite samples across all included sessions/trials, not per session, so class semantics are consistent dataset-wide."* This directly implements the instruction "discretized into five equal percentile bins".

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolated onto the same per-trial `centers` grid as the neural data, so alignment is by construction. The running stream's own timestamps are on the same session clock as the ophys timestamps (both sync-derived), so absolute-time interpolation is valid.

ii.
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
neural_trial  = linear_resample_matrix(ophys_time, events, centers)
```

iii. Step 5 Key Decision 7 (absolute-time alignment for all streams); Step 10 Check 2 verified a raw-reconstructed running trace matched the converted bins exactly; Step 7 plots show *"running and pupil streams smoothly aligned to the trial grid"*.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. From `acquisition/EyeTracking/pupil_tracking/{width, height}` with `acquisition/EyeTracking/eye_tracking/timestamps`. Diameter is defined as `2 · max(width, height)` (major-axis diameter of the fitted pupil ellipse). Blink frames are already stored as `NaN` in these arrays (I confirmed: in `behavior_ophys_experiment_1007107386.nwb`, all 12,500 `likely_blink == True` frames are exactly the 12,500 `NaN` width frames), and those NaNs are filled by time-interpolation. The `likely_blink` dataset itself is not read; the NaN pattern is used instead. Sessions with no `EyeTracking` group at all are dropped.

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

iii. CONVERSION_NOTES.md Step 5 mapping: *"Compute pupil diameter as `2 * max(width, height)` after blink filtering; interpolate onto 30 Hz grid … Exclude sessions with missing eye-tracking acquisition entirely; interpolate within-session for blink-related NaNs"*, referencing the SDK's `process_eye_tracking_data` / `filter_on_blinks` and the whitepaper's eye-tracking description.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) `2·max(width, height)`; (2) NaN gaps (blinks / lost tracking) filled by linear interpolation over the eye-tracking time axis; (3) linear interpolation onto the trial's 30 Hz `centers`; (4) discretisation into 5 global quantile bins. No smoothing or outlier rejection beyond what the Allen pipeline already applied.

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
```

```python
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. Step 5 mapping and Key Decision 11: blink-related NaNs are filled by *"interpolate within-session"* rather than dropped, so every bin has a defined pupil value; sessions with no eye tracking at all are excluded rather than imputed.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same scheme as running speed: five equal-frequency quantile bins with edges computed globally in pass 1 over all finite resampled pupil samples across all kept trials/sessions. Edges found: `[17.08, 73.83, 83.80, 92.87, 105.44, max]`. Achieved distribution exactly 0.200 per class.

ii.
```python
pupil_all = np.concatenate(pupil_values).astype(np.float64, copy=False)
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
...
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. Step 5 Key Decision 10 (global equal-percentile discretisation, shared with running speed); Step 9 verifies *"running and pupil bins were globally balanced at ~0.2 per class."*

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto the same per-trial `centers` grid as the neural data (eye-tracking timestamps are on the same sync-derived session clock), so it is aligned by construction. Raw-file spot checks and the `--show-processing` plots were used to confirm.

ii.
```python
pupil_trial  = linear_resample_vector(pupil_time, pupil_diameter, centers)
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. Step 5 Key Decision 7; Step 10 Check 2 (*"Pupil spot-check: independently compute `2*max(width,height)` from raw eye-tracking arrays, interpolate for a selected trial, and verify `np.allclose()`"* — reported as matching).

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. From the four mutually exclusive boolean columns of the NWB trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, checked in that priority order and mapped to 0/1/2/3. If none is set, the code raises `ValueError` (no fallback category).

ii.
```python
def trial_outcome_index(trial_row: pd.Series) -> int:
    if bool(trial_row["hit"]):
        return 0
    if bool(trial_row["miss"]):
        return 1
    if bool(trial_row["false_alarm"]):
        return 2
    if bool(trial_row["correct_reject"]):
        return 3
    raise ValueError("Trial has no valid outcome label")
```

iii. CONVERSION_NOTES.md Step 1 notes that in `Trial._get_trial_data` *"aborted trials force `go = catch = auto_rewarded = False`; auto-rewarded trials clear hit/miss/false-alarm/correct-reject labels"*, so after the GO/CATCH + non-aborted + non-auto-rewarded filter exactly one of the four flags is always set. Step 5 mapping: *"outcome encoded as constant categorical trace across each trial."*

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The integer code is broadcast to a constant trace across all bins of the trial, so the static per-trial variable is stored in the same `(5, n_timepoints)` time-varying layout as the other four outputs. `output_values` names them `["hit", "miss", "false_alarm", "correct_reject"]`. Full-dataset distribution: hit 0.303, miss 0.571, false_alarm 0.017, correct_reject 0.108.

ii.
```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)

output_trial = np.vstack([
    image_identity,
    image_change,
    running_bin,
    pupil_bin,
    outcome_trace,
])
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 8: *"Represent all outputs as time-varying traces: `image_identity`, `image_change`, `running_speed_bin`, and `pupil_diameter_bin` are naturally time-varying; `trial_outcome` will be repeated across bins within a trial as a constant categorical trace to keep one consistent `(n_output, T)` format."* This follows the instruction's preference *"If at all possible, make it time-varying."*

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Missing eye tracking** (3 active sessions): `KeyError` raised in `get_pupil_data`, caught in pass 1, whole session dropped.
- **Blinks / lost pupil tracking** (NaN in `pupil_tracking`): filled by linear interpolation over time before resampling, so no NaNs reach the discretiser.
- **Omitted stimulus flashes** (5% of repeats): mapped to the `gray` image category rather than left as a spurious image label.
- **Degenerate / duplicate quantile edges**: nudged by machine epsilon so `searchsorted` still yields 5 distinct bins.
- **Empty or inverted trial windows** (`stop_time <= start_time`, non-finite times): `build_trial_bins` returns an empty array and the trial is skipped.
- **Sessions with < 2 trials or no task stimulus block**: dropped in pass 1.
- **All-zero neural trials** (4.84%): retained deliberately, after verifying against the raw NWB that the source `event_detection` segment really is zero.
- **Missing presentation columns**: `read_interval_table` silently skips columns not present in a given file; missing `id` in the trials table falls back to `np.arange(len(trials))`.

Not handled: pass 1 catches only `KeyError` (any other exception aborts the whole run); pass 2 has no `try/except` at all, so `convert_session`'s `RuntimeError` (<2 usable trials) or `trial_outcome_index`'s `ValueError` would abort the full conversion rather than skip the session. Neither fired on this dataset.

ii.
```python
except KeyError as exc:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")
```

```python
if not np.isfinite(start_time) or not np.isfinite(stop_time) or stop_time <= start_time:
    return np.asarray([], dtype=np.float64)
```

```python
values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
```

```python
if np.unique(edges).size < edges.size:
    eps = np.finfo(np.float64).eps
    edges = np.maximum.accumulate(edges + np.arange(edges.size) * eps)
```

```python
def read_interval_table(group: h5py.Group, columns: Iterable[str]) -> pd.DataFrame:
    data = {}
    for column in columns:
        if column not in group:
            continue
        data[column] = read_dataset(group[column])
    return pd.DataFrame(data)
```

iii. CONVERSION_NOTES.md Step 10 Check 5 ("Edge cases"): *"Trial binning uses centers strictly within `[start_time, stop_time)`, avoiding off-by-one inclusion past trial end. Sessions missing `EyeTracking` are excluded up front because pupil output is required. All-zero event trials are retained because they are present in the source data and still have valid behavioral/stimulus labels."* Step 9 quantifies the zero trials and documents the raw-file check that proved they are not a conversion artifact.

## 9-a. What are the most time-consuming steps of the code?

i. The AI identified I/O as the bottleneck and instrumented per-session timing. Measured on the full run (`conversion_full_out.txt`): pass 1 = 91.7 s total (0.46 s/session), pass 2 = 295.0 s total (1.48 s/session), whole conversion 395.6 s for 199 sessions. My own profiling of individual sessions confirms the ranking: reading the `event_detection` matrix from HDF5 dominates and scales with neuron count (0.07 s for a 6-neuron file, 0.68 s for a 208-neuron file), followed by reading the ~4,800-row stimulus presentation table (~0.10–0.14 s) and the per-trial presentation lookup loop (~0.13–0.19 s); the vectorised neural resampling itself is cheap (~0.03 s/session).

ii.
```python
t0 = time.time()
...
print(f"[pass1 {idx}/{len(sessions)}] kept {session.ophys_experiment_id} "
      f"trials={len(trials)} elapsed={time.time() - t0:.2f}s")
...
print(f"[pass2 {idx}/{len(kept_sessions)}] session={session.ophys_experiment_id} "
      f"trials={len(neural_trials)} neurons={brain_region_idx.shape[0]} "
      f"mean_T={mean_t:.1f} elapsed={time.time() - t0:.2f}s")
```

iii. CONVERSION_NOTES.md Step 6: *"Full conversion may still be I/O-heavy because each NWB event matrix must be read from disk. Global binning requires a first pass over sessions, so conversion reads each file twice."* Step 7 gives the estimate (~7.5 min for 202 sessions) that the actual 6.6 min run matched.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI vectorised the one loop it considered important (per-neuron interpolation → `searchsorted` + broadcasting over the whole matrix). Loops that remain and could be vectorised:
- `for trial_idx, trial in trials.iterrows()` (pass 2) and `for trial in trials.itertuples()` (pass 1) — `iterrows()` boxes each row into a Series; `searchsorted` over all trial boundaries at once would replace it.
- `presentations[presentations["trials_id"] == int(trial["id"])]` inside the per-trial loop — a full boolean scan of the ~4,800-row presentation table per trial, i.e. O(n_trials × n_presentations). A single `groupby("trials_id")` (or `np.searchsorted` on sorted presentation start times) would make this O(n). At ~0.13–0.19 s/session this is comparable to the neural resampling cost.
- The inner `for row in trial_presentations.itertuples()` loop, which builds a fresh boolean mask over `centers` per flash (~11 flashes × ~256 bins per trial); a single `np.searchsorted` of `centers` into the flash boundary array would assign all bins at once.
- `decode_str_array`'s per-element Python loop over string columns.

ii.
```python
for trial_idx, trial in trials.iterrows():
    ...
    trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
    for row in trial_presentations.itertuples(index=False):
        mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
        if not mask.any():
            continue
        ...
```

iii. CONVERSION_NOTES.md Step 6 lists only the speed-up that was made (*"Vectorized linear interpolation for neural event matrices using `searchsorted` + broadcasting rather than per-neuron `np.interp`"*) and attributes the remaining cost to I/O. The per-trial presentation scan is not identified anywhere in the notes; the AI stopped optimising once the estimate was comfortably under the 15-minute budget set by the instructions.

## 9-c. What processing does the code repeat multiple times?

i. The two-pass design repeats a substantial amount of work: every kept NWB file is opened **twice**, and in both passes the code re-reads the trials table and the stimulus presentation table, re-reads the running and pupil streams, re-applies `fill_nan_by_time` to the whole-session pupil trace, re-computes `build_trial_bins` for every trial, and re-interpolates running and pupil onto those bins. Pass 1's resampled running/pupil arrays are used only to compute the 10 quantile edges and are then thrown away. That duplicated work is the entire 91.7 s of pass 1. The session metadata table is read once (fine), and the neural event matrix is read only in pass 2 (also fine).

ii.
```python
# pass 1
with h5py.File(session.filepath, "r") as f:
    trials = get_trial_table(f)
    presentations = get_task_presentations(f)
    running_time, running_speed = get_running_data(f)
    pupil_time, pupil_diameter = get_pupil_data(f)
    for trial in trials.itertuples(index=False):
        centers = build_trial_bins(float(trial.start_time), float(trial.stop_time))
        running_trial = linear_resample_vector(running_time, running_speed, centers)
        pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
        running_values.append(running_trial); pupil_values.append(pupil_trial)

# pass 2 — same four reads, same bins, same two interpolations, all recomputed
with h5py.File(session.filepath, "r") as f:
    trials = get_trial_table(f)
    presentations = get_task_presentations(f)
    ophys_time, events = get_neural_data(f)
    running_time, running_speed = get_running_data(f)
    pupil_time, pupil_diameter = get_pupil_data(f)
    for trial_idx, trial in trials.iterrows():
        centers = build_trial_bins(...)
        running_trial = linear_resample_vector(running_time, running_speed, centers)
        pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. CONVERSION_NOTES.md Step 6 acknowledges it: *"Global binning requires a first pass over sessions, so conversion reads each file twice"*, and justifies the design as *"Session-level streaming design avoids storing continuous raw traces for the whole dataset in memory."* (In practice the memory argument is weak, since pass 2 still accumulates all 8.1 GB of neural trials in RAM before pickling; caching only the per-trial running/pupil vectors from pass 1, as the reference does, would have cost little memory and saved the whole pass.)

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Work performed whose result is never used:
- `get_trial_table` reads, casts and stores `initial_image_name` and `change_image_name` for every trial; the converted outputs are built entirely from the stimulus presentation table, so these two columns are never consumed.
- `get_task_presentations` reads `active` and `duration` columns and rebuilds `stimulus_block_name` as a DataFrame column; none is used downstream.
- Pass 1 materialises the full resampled running and pupil trace for every trial of every session (~51k trials × ~256 bins × 2) purely to feed `np.quantile`, then discards all of it (see 9-c).
- `get_pupil_data` computes `2·max(width, height)` and NaN-fills the entire ~136k-sample session trace even though only the bins inside kept trials are ever sampled.
- `get_pupil_data` reads the full `height` array (only used via the element-wise max) and `eye_tracking/timestamps` for the whole session.
- An `input` array of shape `(0, T)` is allocated per trial (51,075 empty arrays) although the task specifies no decoder inputs; the reference does the same with `(0,)`.
- Unused imports (`math`, `Iterable` is used, `math` is not) and an unused `trial_idx` loop variable.

None of this is scientifically harmful; it costs roughly the 92 s of pass 1 plus a small constant per session.

ii.
```python
for column in ["initial_image_name", "change_image_name"]:
    if column in trials:
        trials[column] = trials[column].astype(str)   # never read afterwards
```

```python
columns = ["start_time", "stop_time", "image_name", "omitted", "is_change",
           "trials_id", "stimulus_block_name", "active", "duration"]   # 'active', 'duration' unused
```

```python
diameter = 2.0 * np.maximum(width, height)
diameter = fill_nan_by_time(timestamps, diameter)     # whole session, only trial bins used
```

```python
input_trials = [np.zeros((0, trial.shape[1]), dtype=np.float32) for trial in neural_trials]
```

iii. CONVERSION_NOTES.md does not flag any of these as waste. The trials-table image columns were read as part of the Step 5 plan to *"use trial table to sanity-check change times and image names"*, but that cross-check ended up being done in the separate sanity-check script (`cache/raw_sanity_checks.py`) and in the diagnostic plots rather than in `convert_data.py`, leaving the reads vestigial.
