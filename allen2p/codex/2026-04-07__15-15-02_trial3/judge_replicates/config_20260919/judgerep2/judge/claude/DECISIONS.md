# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK `VisualBehaviorOphysProjectCache`. It first tried `pynwb.NWBHDF5IO` and `BehaviorOphysExperiment.from_nwb_path()`, both of which failed in this environment with an HDMF/pynwb cached-namespace version conflict (`core - cached version: 2.6.0-alpha, loaded version: 2.7.0`). It therefore reads the released NWB files **directly with `h5py`** and reads the project metadata from the shipped CSV manifest.

Concretely:
- Session inventory comes from `data/visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv`, intersected with the `behavior_ophys_experiment_*.nwb` files actually present on disk (284 files).
- Rows with `passive == True` are dropped, leaving 202 active experiments.
- **No `project_code` filter is applied**, so both `VisualBehavior` (168 single-plane, ~30.9 Hz) and `VisualBehaviorMultiscope` (34 planes, ~10.7 Hz) experiments are included.
- Per file, data are pulled from fixed NWB paths: `intervals/trials`, `intervals/*_presentations`, `processing/ophys/event_detection/{data,timestamps}`, `processing/running/speed`, `acquisition/EyeTracking/*`, `processing/ophys/image_segmentation/cell_specimen_table`.
- The pipeline is **two-pass**: pass 1 re-opens every file to collect global running/pupil percentile edges and to apply session-level QC; pass 2 re-opens the surviving files and builds the trial matrices.

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
print("Pass 1: collecting global running/pupil statistics")
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
print("Pass 2: converting sessions")
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. From `CONVERSION_NOTES.md` Step 4: *"Local environment cannot instantiate `NWBFile` for these NWBs due `external_resources`/version mismatch … Read NWB files directly with `h5py` and mirror the AllenSDK/whitepaper semantics from the processed NWB contents rather than relying on broken high-level loading in this environment."* The AI argues the NWB files already contain the Allen-processed tables, so reading them directly uses the same processed sources as the SDK, only with a different loader. For scope it argues (Step 4): *"Use the broader active-task release, not the paper's strategy-specific subset … Passive sessions are not task performance and make outcome labels degenerate."*

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the sessions that survive QC, sorted as strings. `subject_idx` maps each converted session to its mouse. 38 mice result.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. `mouse_id` is taken from `ophys_experiment_table.csv`, described in the notes as the canonical animal identifier; the Step 10 check re-derived 38 subjects directly from the raw metadata and matched the converted file.

## 1-c. How are the data split into sessions?

i. **Each `ophys_experiment_id` (i.e. each imaging plane) is treated as one converted "session."** The AI explicitly noticed that several experiment files share an `ophys_session_id` (the Multiscope recordings: 3 sessions of 7 planes, 2 of 5, 1 of 3) and deliberately chose *not* to merge them. The consequence is that one real Multiscope session becomes up to 7 separate entries in `data['neural']`, each with its own neuron set but carrying the *same* behavioral trials. For mouse `457841` this yields 34 "sessions" from 6 real recordings. Final counts: 202 active experiments → 199 converted "sessions" (174 distinct `ophys_session_id`s).

ii.
```python
sessions.append(
    SessionMeta(
        ophys_experiment_id=int(row.ophys_experiment_id),
        ophys_session_id=int(row.ophys_session_id),
        ...
    )
)
```
```python
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(session=session, ...)
    neural_all.append(neural_trials)
```
Session identity is only recorded in metadata, never used to group:
```python
"included_ophys_session_ids": [session.ophys_session_id for session in kept_sessions],
```

iii. `CONVERSION_NOTES.md` Step 5, Key Decision 2: *"Treat each `ophys_experiment_id` file as one converted session: This matches the AllenSDK object granularity (`BehaviorOphysExperiment`) and yields a single imaging plane / neuron set / brain region per session."* Step 10 Check 5 repeats it: *"Multiple experiment files can share the same `behavior_session_id` or `ophys_session_id`; the conversion intentionally treats each `ophys_experiment_id` plane as a separate session because that is the AllenSDK experiment granularity and each file has its own neuron set."*

## 1-d. How are the data split into trials?

i. Trials come from the Allen-processed `intervals/trials` table inside each NWB. Kept trials are `(go | catch) & ~aborted & ~auto_rewarded`. Each trial spans the full `start_time` → `stop_time` window (mean ≈ 8.5 s, range ≈ 7.3–12.5 s), so trials are variable length (211–377 bins). Bin centres are laid on a fixed 30 Hz grid inside that window.

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

iii. Step 1 notes that `trial_masks.contingent_trials` in the SDK defines contingent trials as GO + CATCH only, and that `Trial._get_trial_data` forces `go = catch = False` for aborted trials and clears outcome flags for auto-rewarded trials. Step 5, Key Decision 3: *"Segment trials using the processed NWB `trials` table … mirrors the Allen reference processing without re-deriving trial logic from lower-level files."* Key Decision 4: *"Keep only contingent trials: Include GO and CATCH trials; exclude `aborted` and `auto_rewarded` exactly as required."*

## 1-e. How are trials filtered based on quality controls?

i. Filtering happens at several levels:
- **Session level (before trials):** passive experiments dropped; experiments with no `acquisition/EyeTracking` group dropped (`KeyError` caught in pass 1 — 3 sessions); experiments with fewer than 2 kept trials dropped; experiments with no `change_detection` stimulus block dropped.
- **Trial level:** only `(go | catch) & ~aborted & ~auto_rewarded`; trials whose bin grid is empty (`centers.size == 0`, i.e. `stop_time <= start_time` or non-finite) are skipped.
- **Retained despite a warning:** 2,474 / 51,075 (4.84 %) trials have an all-zero event trace. The AI checked the raw NWB and confirmed the source `event_detection` block is genuinely zero, then deliberately kept them.
- Final yield: 199 sessions, 38 mice, 51,075 trials, 29,168 neurons.

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
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
if centers.size == 0:
    continue
...
if len(neural_trials) < 2:
    raise RuntimeError(f"Session {session.ophys_experiment_id} has fewer than 2 usable trials")
```

iii. Step 5, Key Decision 11: *"Require pupil availability at session level: Three active local files lack eye-tracking acquisition entirely; these sessions will be excluded."* Step 10: *"All-zero event trials are retained because they are present in the source data and still have valid behavioral/stimulus labels."* Step 10 Check 4 recomputed `(go | catch) & ~aborted & ~auto_rewarded` directly from the raw tables and obtained exactly 51,075, matching the converted file.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` matrix is the **detected-calcium-event magnitude trace**, read from `processing/ophys/event_detection/data` (shape `time × ROI`) with its own `timestamps`. dF/F (`processing/ophys/dff`) is explicitly *not* used, nor are the SDK's `filtered_events`.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```
```python
"neural_signal": "ophys event magnitudes from NWB event_detection",
```

iii. Step 4: *"Paper methods explicitly use detected calcium events for neural analyses … Use raw event magnitude traces from `processing/ophys/event_detection/data` as `neural`. Do not use `filtered_events` (visualization-only) and do not recompute dF/F."* This is backed by `methods.txt:208`: *"For all analysis of neural data we used the detected calcium events as described in Garrett et al."* and `methods.txt:179`: *"We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces."* The whitepaper's FastLZero event-detection section is cited for provenance.

## 2-b. How is the `neural` data processed?

i. The only processing is **transposition and linear resampling onto the fixed 30 Hz trial grid**. No normalisation, smoothing, z-scoring, binning/averaging, or cross-plane merging is applied. Resampling is a vectorised two-point linear interpolation over all ROIs at once (`searchsorted` + broadcasting), with `idx_hi` clipped to `[1, len-1]` so out-of-range bin centres are linearly extrapolated. The result is stored as `float32` with shape `(n_neurons, n_bins)`.

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

iii. Step 6: *"Vectorized linear interpolation for neural event matrices using `searchsorted` + broadcasting rather than per-neuron `np.interp`."* Step 4 justifies the common grid: *"Papers/whitepaper describe mixed acquisition rates and also describe 30 Hz interpolation for event-triggered analyses. Resample all streams to a common 30 Hz grid … This preserves ophys-based timing while satisfying the decoder requirement that all trials/sessions share a common bin size."* The events themselves are taken as-is because the whitepaper pipeline already produced them.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neuron-level filtering is applied in the conversion code.** All ROIs in `processing/ophys/image_segmentation/cell_specimen_table` are kept, and `brain_region_idx` is simply a constant-region vector of that length. The AI justified this by checking that the released NWBs contain only ROIs already passing the Allen ROI-filtering step: it reported `29,168 valid of 29,168 total listed`. (I independently confirmed `valid_roi` is all-`True` in 46 sampled experiment files.)

ii.
```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

iii. Step 10, Check 3: *"reference: `CellSpecimens.__init__` keeps `valid_roi == True`; converter: included-session raw NWB files already had all listed cells valid (29,168 total valid of 29,168 total listed), so event matrices matched converted neuron counts exactly."* Step 3 records the whitepaper ROI-filtering rules (motion border, duplicate/union, dendrite, too small/narrow/dim) as already applied upstream.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start**. For each trial the grid is `start_time + (k + 0.5)·(1/30)` for all bins whose centre falls before `stop_time`; all streams (neural, running, pupil, stimulus) are evaluated on that same absolute-time grid, so alignment across streams is exact by construction. Metadata records `temporal_alignment_event = "trial start"`, `off_start = 0.0`, `off_end = None` (trials are variable length). Neural values at each bin centre come from linear interpolation of the ophys event trace — no index rounding, so there is no half-frame quantisation offset.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
if centers.size == 0:
    continue

neural_trial = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```
```python
"temporal_alignment_event": "trial start",
"off_start": 0.0,
"off_end": None,
```

iii. Step 5, Key Decision 7: *"Align by absolute ophys time, then cut into trials: For each trial, create bin centers from trial `start_time` to `stop_time` at 30 Hz and sample/interpolate all streams onto that grid."* Step 10, Check 2 reports raw-NWB `np.allclose()` spot checks on three trials (experiments `775614751`, `792815735`, `960410028`) with max abs diffs of `2.98e-08`, `0.0`, `1.19e-07`, and all output rows matching exactly. The `--show-processing` plots (`processing_775614751.png`, `processing_788490510.png`) overlay raw vs. resampled traces and the change-flash marker as a visual alignment check.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is a **fixed 33.333 ms (30 Hz) for every trial and every session**, stored as `time_bin_size = 33.333…`. Rebinning **is** applied: every stream, including neural, is linearly resampled from its native clock onto this grid. This is a mild *down*-sampling for the 168 single-plane `VisualBehavior` experiments (native ≈ 30.94 Hz) and roughly a **3× up-sampling** for the 34 `VisualBehaviorMultiscope` planes (native ≈ 10.73 Hz). Resampling is interpolation, not binning/averaging or summing — no accumulation of event magnitude is performed. Resulting trial lengths are 211–377 bins (mean 254), and the full pickle is 8.6 GB.

ii.
```python
DT = 1.0 / 30.0
TIME_BIN_MS = DT * 1000.0
...
n_bins = max(1, int(np.ceil((stop_time - start_time) / DT)))
centers = start_time + (np.arange(n_bins, dtype=np.float64) + 0.5) * DT
```
```python
"time_bin_size": TIME_BIN_MS,
"sampling_grid_hz": 30.0,
```

iii. Step 4: *"Native sampling rates … Resample all streams to a common 30 Hz grid (`33.333... ms` bins). This preserves ophys-based timing while satisfying the decoder requirement that all trials/sessions share a common bin size."* Step 5, Key Decision 6: *"Native acquisition rates vary across rigs (31 Hz single-plane, 11 Hz multiplane). Resampling all streams to 30 Hz gives one shared bin size while remaining close to behavior/eye-tracking rate and consistent with paper event-triggered interpolation onto 30 Hz timestamps."*

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is built from the **stimulus-presentation interval tables**, not from the trial table. `get_task_presentations` walks every group under `intervals/` except `trials`, keeps rows whose `stimulus_block_name` contains `change_detection`, and pulls `start_time`, `stop_time`, `image_name`, `omitted`, `is_change`, `trials_id`. Every bin whose centre falls inside a non-omitted flash gets that flash's `image_name`; all other bins (inter-stimulus grey, omitted flashes, pre/post-flash padding) get an explicit `gray` category. The trial table's `initial_image_name` / `change_image_name` are read but only used as a cross-check, never to build the output.

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

iii. Step 5 mapping table: *"Build per-bin categorical state on 30 Hz grid: actual `image_name` during image flashes; `gray` during gray/omitted periods … Restrict to `stimulus_block_name == change_detection_behavior` when present."* Key Decision 9: *"Encode gray/omission periods explicitly: `image_identity` will include a `gray` category for ISI and omitted-image periods, because the user specifically asks for the image identity during non-gray screen and those periods still occupy trial time."* Key Decision 12 explains restricting to the task block so natural-movie / spontaneous blocks are ignored.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Image names are collected across all pass-1 sessions into a global set seeded with `gray`, sorted, then `gray` is forced to index 0 and the remaining 16 image names follow alphabetically. The per-bin string trace is mapped to this global integer code. `output_values[0]` stores the ordered names. Result: 17 categories, with `gray` occupying 67.0 % of all bins (matching the 250 ms-on / 500 ms-off duty cycle plus 5 % omissions).

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
"output_values": [
    image_values,
    ...
```

iii. A global, deterministic mapping is used so that codes are comparable across sessions; `gray` is pinned to 0 so the "no image" state is the natural baseline class. The Step 9 verification confirms `image_identity` range `[0, 16]` and a per-image fraction of ≈ 0.019–0.022 each with `gray` at 0.670.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed on exactly the same `centers` array used for the neural matrix, by testing each bin centre against the flash's `[start_time, stop_time)` interval. Because both are evaluated in absolute session time on one shared grid, they are aligned by construction with zero relative offset. Presentations are additionally restricted to the ones whose `trials_id` equals the trial's `id`.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
...
image_identity[mask] = image_value_to_idx[str(row.image_name)]
...
output_trial = np.vstack([image_identity, image_change, running_bin, pupil_bin, outcome_trace])
neural_trials.append(neural_trial.astype(np.float32, copy=False))
output_trials.append(output_trial)
```

iii. Step 5, Key Decision 7 (shared 30 Hz absolute-time grid for all streams). Step 10, Check 2 confirms the image trace of three trials reconstructed independently from the raw interval tables matched the converted arrays exactly, and Check 5 notes: *"Trial binning uses centers strictly within `[start_time, stop_time)`, avoiding off-by-one inclusion past trial end."*

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the `is_change` boolean column of the same task stimulus-presentation table (together with that presentation's `start_time` / `stop_time`). The trial table's `change_time` and `go` columns are *not* used to build it. `is_change` is true only where the flashed image actually differs from the previous one, so catch (sham-change) trials naturally get an all-zero trace — I verified on experiment `1007107386` that all 323 go trials have exactly one `is_change` flash and all 42 catch trials have none.

ii.
```python
columns = [
    "start_time", "stop_time", "image_name", "omitted", "is_change",
    "trials_id", "stimulus_block_name", "active", "duration",
]
df = read_interval_table(group, columns)
...
out["is_change"] = out["is_change"].fillna(0).astype(bool)
```
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. Step 1 identifies `get_stimulus_presentations` / `is_change_event` in the SDK as the functions that *"mark image identity changes from successive non-omitted images."* Step 5 mapping: *"Binary per-bin trace: 1 during change-image flash interval, else 0 … Change and pre-change flashes are never omitted."*

## 4-b. What processing is involved in computing `output` *Image change*?

i. None beyond the interval test: the indicator is set to 1 for the bins whose centres fall inside the **change flash itself**, i.e. a ~250 ms window (≈ 7–8 bins at 30 Hz). It is *not* extended over the following grey inter-stimulus interval, and it is not extended over the whole post-change period. Across the full dataset this gives `{no_change: 0.975, change: 0.025}`.

ii.
```python
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    if not mask.any():
        continue
    ...
    if bool(row.is_change):
        image_change[mask] = 1
```
```python
"output_values": [..., ["no_change", "change"], ...]
```

iii. Step 5 mapping table states the indicator marks the change-image flash interval only. The instruction quoted in the notes is *"Have value of 1 right after a change in image identity, otherwise 0"*, which the AI reads as the change flash itself. Step 12 notes the resulting sparsity (*"image change has sparse positives (2.5% of bins)"*) and argues balanced accuracy compensates.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is required — the variable is natively binary (`is_change`), encoded as `{0: no_change, 1: change}`, `int64`.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
image_change[mask] = 1
```

iii. Not applicable; the AI treats `is_change` as an already-categorical source flag from the Allen processing (Step 1: `is_change_event`).

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity: the same `centers` grid, the same `[start_time, stop_time)` interval test, the same `trials_id` restriction. It is emitted as row 1 of the `(5, n_bins)` output matrix, so it is bin-for-bin aligned with the neural matrix.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
if bool(row.is_change):
    image_change[mask] = 1
...
output_trial = np.vstack([image_identity, image_change, running_bin, pupil_bin, outcome_trace])
```

iii. Same rationale as 3-c — one shared absolute-time grid for every stream (Step 5, Key Decision 7). Step 10, Check 2 confirmed `image_change` matched independent raw-data reconstruction on three trials, and the `--show-processing` plots draw the change indicator against the change flash and the trial's `change_time` line.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `processing/running/speed` — the AllenSDK's low-pass-**filtered** running speed in cm/s, together with its own timestamps. The unfiltered `speed_unfiltered` and the raw wheel `dx` are not used.

ii.
```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. Step 1 identifies `RunningSpeed.from_stimulus_file` / `_get_running_speed_df`, noting the SDK *"Computes running speed from wheel signals on stimulus timestamps with no monitor delay and optional low-pass filtering."* Step 5 mapping: *"Use filtered running speed in cm/s."*

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation from the ~60 Hz running clock onto the 30 Hz trial grid (via `np.interp`, which clamps at the ends rather than producing NaN), then discretisation into 5 global equal-frequency bins. No smoothing, clipping of negative speeds, or per-session normalisation.

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

iii. Step 5 mapping: *"Interpolate filtered running speed onto 30 Hz trial grid; discretize globally across included data into 5 equal-frequency bins."* The 30 Hz target is close to the native behaviour rate, so interpolation is near-lossless.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five **equal-percentile (quintile) bins**. Edges are `np.quantile` at `[0, 0.2, …, 1.0]` computed in pass 1 over **all finite running samples pooled from every kept trial of every kept session** — i.e. one global set of edges, not per-session. Values are clipped to `[edges[0], edges[-1]]` and assigned by `searchsorted` on the interior edges, giving codes 0–4. Duplicate edges (e.g. a mass of zeros at rest) are nudged apart by machine-epsilon accumulation so the bins stay strictly ordered. The verification output shows the realised distribution is exactly `{q1: 0.200, q2: 0.200, q3: 0.200, q4: 0.200, q5: 0.200}`.

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

iii. Step 5, Key Decision 10: *"Discretize running and pupil globally across the included dataset: Five equal-percentile bins will be computed from all finite samples across all included sessions/trials, not per session, so class semantics are consistent dataset-wide."* The edges are also written to `metadata['running_bin_edges']` for recovery.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Same `centers` grid as the neural matrix; the interpolation is evaluated at the identical absolute times, so the running row of the output matrix is bin-for-bin aligned with the neural matrix. No monitor-delay or extra offset is added (consistent with the SDK, which applies zero monitor delay to running timestamps).

ii.
```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
...
output_trial = np.vstack([image_identity, image_change, running_bin, pupil_bin, outcome_trace])
```

iii. Step 1 explicitly notes that *"running speed timestamps explicitly require zero monitor delay"*, so no correction is applied. Step 10, Check 2 verified the pre-discretised running trace against an independent raw-data interpolation with `np.allclose()` for three trials.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. From `acquisition/EyeTracking/pupil_tracking/{width, height}` plus `acquisition/EyeTracking/eye_tracking/timestamps`. Diameter is defined as `2 × max(width, height)`, i.e. the major axis of the fitted pupil ellipse. The `likely_blink` dataset is **not read explicitly**; the AI relies on blink frames being already `NaN` in the released `pupil_tracking` arrays. (I confirmed this holds: in three sampled files the NaN fraction of `2·max(w,h)` equals the `likely_blink` fraction exactly — 0.092, 0.029, 0.053 — and every blink frame is NaN, because the AllenSDK's `filter_on_blinks` was applied before NWB write.) Sessions with no `EyeTracking` group at all are dropped.

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

iii. Step 5 mapping: *"Compute pupil diameter as `2 * max(width, height)` after blink filtering; interpolate onto 30 Hz grid; discretize globally into 5 equal-frequency bins … Exclude sessions with missing eye-tracking acquisition entirely; interpolate within-session for blink-related NaNs."* Step 1 cites `process_eye_tracking_data` / `filter_on_blinks` as the SDK provenance for the blink handling.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Three stages: (1) `2·max(width, height)`; (2) `fill_nan_by_time` — every NaN sample (blinks and lost frames) is replaced by linear interpolation in time from the surrounding finite samples, with `np.interp`'s end-clamping for leading/trailing NaNs, so the session trace has no NaNs left; (3) linear resampling onto the 30 Hz trial grid and discretisation into 5 global quintile bins (same machinery as running speed). Because NaNs are already removed, no bin is ever assigned by a NaN fallback.

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

iii. Step 5, Key Decision 11: *"Remaining sessions have modest blink-related missingness and can be filled by time interpolation before discretization."* Step 10, Check 2 verified the pupil trace against an independent raw reconstruction (`2*max(width,height)` recomputed from raw arrays) with `np.allclose()`.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five global equal-percentile bins, exactly the same procedure and code path as running speed: edges from `np.quantile` over all finite pupil samples pooled across every kept trial of every kept session in pass 1, then `clip` + `searchsorted`. Realised distribution in the full dataset is exactly `{q1: 0.200, …, q5: 0.200}`; edges are saved as `metadata['pupil_bin_edges']`.

ii.
```python
pupil_all = np.concatenate(pupil_values).astype(np.float64, copy=False)
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
...
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```
```python
"pupil_bin_edges": pupil_edges.tolist(),
```

iii. Step 5, Key Decision 10 (global rather than per-session edges, so class semantics are dataset-wide consistent). Planned sanity check: *"Distribution checks: global percentile bins for running/pupil produce approximately balanced class counts."*

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same `centers` grid, same absolute-time interpolation as the neural stream; emitted as row 3 of the `(5, n_bins)` output matrix. The AI assumes (as the SDK does) that eye-tracking timestamps are already on the same synced hardware clock as the ophys frames, so no additional offset is applied.

ii.
```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
...
output_trial = np.vstack([image_identity, image_change, running_bin, pupil_bin, outcome_trace])
```

iii. Step 5, Key Decision 7. Step 1 notes `EyeTrackingTable.from_data_file` *"Aligns eye-tracking frames to timestamps"* upstream, so the stored timestamps are already sync-derived.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. From the four mutually exclusive boolean columns of the NWB trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, checked in that fixed order. Since aborted and auto-rewarded trials were already removed, exactly one flag is set per kept trial; if none is set the code raises rather than silently emitting a sentinel.

ii.
```python
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]

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

iii. Step 1: *"Within `Trial._get_trial_data`, aborted trials force `go = catch = auto_rewarded = False`; auto-rewarded trials clear hit/miss/false-alarm/correct-reject labels"* — hence the four flags are exhaustive and exclusive precisely on the kept subset. Step 5 mapping: *"outcome encoded as constant categorical trace across each trial."*

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The integer code 0–3 is broadcast to a constant vector over all bins of the trial, so the per-trial static variable is delivered in the same `(n_output, n_timepoints)` time-varying layout as the other four outputs. `output_values[4] = ["hit", "miss", "false_alarm", "correct_reject"]`. Full-dataset distribution: hit 0.303, miss 0.571, false_alarm 0.017, correct_reject 0.108.

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

iii. Step 5, Key Decision 8: *"Represent all outputs as time-varying traces … `trial_outcome` will be repeated across bins within a trial as a constant categorical trace to keep one consistent `(n_output, T)` format."* This follows the format spec's preference *"If at all possible, make it time-varying."*

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Missing eye tracking** (3 active experiments): `get_pupil_data` raises `KeyError`, caught in pass 1, session skipped entirely.
- **Missing / no task stimulus block**: `get_task_presentations` returns an empty frame → session skipped.
- **Blink / lost-frame NaNs in pupil**: filled by in-time linear interpolation (`fill_nan_by_time`), with end-clamping for leading/trailing NaNs.
- **Missing columns in an interval table**: `read_interval_table` silently skips absent columns; missing `id` falls back to `np.arange`; missing `omitted`/`is_change` are `fillna(0)`.
- **Byte vs. str NWB string arrays**: `decode_str_array` normalises both.
- **Degenerate trial windows** (`stop_time <= start_time`, non-finite times): `build_trial_bins` returns an empty array and the trial is skipped.
- **Degenerate quantile edges** (ties, e.g. a large mass of zero running speed): nudged apart by epsilon accumulation.
- **Sessions with < 2 usable trials**: skipped in pass 1 (and `RuntimeError` in pass 2).
- **All-zero neural trials** (4.84 %): deliberately retained after raw-data confirmation.

Gaps: pass 1 catches only `KeyError`, so a `ValueError` from `fill_nan_by_time` (all-NaN pupil) would abort the run; pass 2 has no `try/except` at all, so an error there (including the `RuntimeError` for < 2 usable trials, which pass 1 cannot anticipate because it counts table rows rather than non-empty bin grids) would terminate the whole conversion.

ii.
```python
except KeyError as exc:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")
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
```python
out["omitted"] = out["omitted"].fillna(0).astype(bool)
out["is_change"] = out["is_change"].fillna(0).astype(bool)
```
```python
if np.unique(edges).size < edges.size:
    eps = np.finfo(np.float64).eps
    edges = np.maximum.accumulate(edges + np.arange(edges.size) * eps)
```

iii. Step 5, Key Decision 11 (drop sessions with no eye tracking, interpolate blink NaNs). Step 10: *"All-zero event trials are retained because they are present in the source data and still have valid behavioral/stimulus labels."* and *"Raw-NWB spot check on warned session index 3 (experiment 792815735) … showed the underlying `event_detection` segment itself was exactly zero across all 27 neurons and 388 native frames. This indicates the warnings are due to sparse event detections in the source data rather than a conversion bug."*

## 9-a. What are the most time-consuming steps of the code?

i. Disk I/O on the NWB files dominates. The full run took **395.6 s** total: pass 1 ≈ 0.43–0.48 s per file × 202 files ≈ 91 s, pass 2 ≈ 1.5 s per file × 199 files ≈ 300 s. Within pass 2 the costs are reading the full-session `event_detection` matrix (e.g. 140,204 × 13 to 140,204 × 666 float32), the eye-tracking arrays (~136 k samples), and the running trace (~270 k samples), plus the per-trial × per-neuron interpolation. Because the design is two-pass, **every file is opened and partially read twice**. Pickling the 8.6 GB result is also a non-trivial tail cost.

ii.
```python
t0 = time.time()
...
print(
    f"[pass2 {idx}/{len(kept_sessions)}] session={session.ophys_experiment_id} "
    f"trials={len(neural_trials)} neurons={brain_region_idx.shape[0]} "
    f"mean_T={mean_t:.1f} elapsed={time.time() - t0:.2f}s"
)
```

iii. Step 6: *"Full conversion may still be I/O-heavy because each NWB event matrix must be read from disk. Global binning requires a first pass over sessions, so conversion reads each file twice."* Step 7 estimated ~7.5 min for 202 sessions from the 2-session sample, and the actual 6.6 min run confirmed that estimate (within the required 1.5× tolerance, so no re-optimisation was triggered).

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorised the single hottest inner loop — neural resampling is done for all ROIs at once instead of per neuron. What remains loop-based:
- The **per-trial Python loop** in `convert_session` (51,075 iterations) and its mirror in `collect_global_statistics`. All of `build_trial_bins`, the three resampling calls, and the two `digitize_with_edges` calls could be done once per session on a whole-session 30 Hz grid, with trials then extracted by slicing — this would replace ~51 k small interpolations with ~199 large ones.
- The **nested per-presentation loop** (`for row in trial_presentations.itertuples()`), roughly 11 flashes × 51 k trials ≈ 560 k iterations, each doing a full boolean comparison over the trial's ~254 bin centres. This is the most clearly vectorisable remaining loop: a single `np.searchsorted` of the session grid into the presentation start/stop edges would assign every bin's image code in one shot.
- `decode_str_array` is a pure Python loop over every string element of every interval column read (including `stimulus_block_name`, ~4,800 rows per session, read for every group under `intervals/`).
- The per-session `itertuples` loop in `collect_global_statistics` duplicates work that pass 2 redoes.

ii.
```python
for trial_idx, trial in trials.iterrows():
    centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
    ...
    trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
    for row in trial_presentations.itertuples(index=False):
        mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
```
```python
def decode_str_array(values: np.ndarray) -> np.ndarray:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        ...
```

iii. Step 6 records the one vectorisation the AI did: *"Vectorized linear interpolation for neural event matrices using `searchsorted` + broadcasting rather than per-neuron `np.interp`."* The remaining loops were not called out as issues; Step 7's timing analysis concluded the estimated full runtime (~7.5 min) was already under the 15-minute threshold, so no further optimisation was pursued.

## 9-c. What processing does the code repeat multiple times?

i. The two-pass design repeats a substantial amount of work per session:
- `h5py.File` open, `get_trial_table`, and the `(go|catch) & ~aborted & ~auto_rewarded` filter run **twice** per session.
- `get_task_presentations` (which walks every `intervals/` group and string-decodes `stimulus_block_name`) runs **twice**.
- `get_running_data` and `get_pupil_data` (including the whole-session `fill_nan_by_time`) run **twice**.
- `build_trial_bins` and the running/pupil per-trial interpolation run **twice** for all 51,075 trials — once to gather quantile statistics, once to emit the output.
- Within pass 2, `get_cell_count_and_region_idx` and the session-level stream reads are correctly done once per session, not per trial.

ii.
```python
# pass 1
with h5py.File(session.filepath, "r") as f:
    trials = get_trial_table(f)
    trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
    presentations = get_task_presentations(f)
    running_time, running_speed = get_running_data(f)
    pupil_time, pupil_diameter = get_pupil_data(f)
    for trial in trials.itertuples(index=False):
        centers = build_trial_bins(float(trial.start_time), float(trial.stop_time))
        running_trial = linear_resample_vector(running_time, running_speed, centers)
        pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```
```python
# pass 2 — same four loaders, same trial filter, same bin construction
with h5py.File(session.filepath, "r") as f:
    trials = get_trial_table(f)
    trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
    presentations = get_task_presentations(f)
    ophys_time, events = get_neural_data(f)
    running_time, running_speed = get_running_data(f)
    pupil_time, pupil_diameter = get_pupil_data(f)
```

iii. Step 6 acknowledges the trade-off: *"Global binning requires a first pass over sessions, so conversion reads each file twice"*, and justifies it as a memory decision: *"Session-level streaming design avoids storing continuous raw traces for the whole dataset in memory."* The alternative (cache the pass-1 running/pupil trial arrays, as the reference solution does) would have eliminated the duplication at the cost of holding ~51 k small arrays in RAM.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Work that is computed and then thrown away:
- **Pass 1's per-trial running and pupil arrays** (all 51,075 trials × 2 streams) are concatenated only to compute ten quantile edges, then discarded; pass 2 recomputes them from scratch.
- **Pass 1's `get_task_presentations` call** is used only to test emptiness and to harvest the set of image names — the parsed table itself is dropped.
- **Unused columns** are read and string-decoded on every session: `active` and `duration` in the presentations table, and `catch` / `stop_time` / `change_time` / `initial_image_name` / `change_image_name` in the trials table are read but never used to build any output (`change_time` and the image-name columns are used only in the optional `--show-processing` plot).
- **Up-sampling the 34 Multiscope planes from 10.7 Hz to 30 Hz** roughly triples their stored size without adding information; combined with `float32` storage this is a major contributor to the 8.6 GB output pickle.
- **`input_trials`**: an empty `(0, T)` array is allocated per trial for all 51,075 trials even though `input_names` is empty and the decoder uses no inputs.
- **`SessionMeta` fields** `behavior_session_id`, `session_type`, `experience_level`, `project_code` are carried through the pipeline but only two of them reach the metadata dict.

ii.
```python
running_values.append(running_trial)
pupil_values.append(pupil_trial)
...
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
return running_edges, pupil_edges, kept_sessions, image_names   # trial arrays dropped
```
```python
columns = [
    "start_time", "stop_time", "image_name", "omitted", "is_change",
    "trials_id", "stimulus_block_name", "active", "duration",
]
```
```python
input_trials = [np.zeros((0, trial.shape[1]), dtype=np.float32) for trial in neural_trials]
```

iii. The notes do not flag these as unnecessary. Step 6 frames the double work as an intentional memory trade-off (*"Session-level streaming design avoids storing continuous raw traces for the whole dataset in memory"*), and Step 5, Key Decision 6 justifies the 30 Hz grid on format-consistency grounds rather than discussing its storage cost. Step 5 mapping explains the empty inputs: *"Use empty arrays of shape `(0, n_timepoints)` for every trial … Decoder task specifies no decoder inputs."*
