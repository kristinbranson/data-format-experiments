# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. **Decisions**

- The AI does **not** use the AllenSDK object model at all. After two failed attempts to open a
  local NWB (`pynwb.NWBHDF5IO(..., load_namespaces=True)` without importing `allensdk` first, and
  `BehaviorOphysExperiment.from_nwb_path` run with `PYTHONPATH=/app/code`, which shadows the
  installed `allensdk` with the repo copy), it concluded that "the local AllenSDK/NWB stack cannot
  instantiate these NWB files" and decided to read the processed NWB/HDF5 groups directly with
  `h5py`.
- The master listing of data is still the SDK's canonical manifest CSV:
  `data/visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv`.
- The table is intersected with the NWB files actually present on disk
  (`behavior_ophys_experiment_<ophys_experiment_id>.nwb`, 284 files), then filtered to
  `passive == False` (202 files), then sorted by `ophys_experiment_id`.
- **No `project_code` filter is applied**, so both `VisualBehavior` (168 active) and
  `VisualBehaviorMultiscope` (34 active) experiments are loaded.
- Per file, the AI reads only the processed groups it needs:
  `processing/ophys/event_detection` (neural + ophys timestamps),
  `processing/running/speed`, `acquisition/EyeTracking/{pupil_tracking,eye_tracking}`,
  `intervals/trials`, and the `*_presentations` interval tables whose
  `stimulus_block_name` contains `change_detection`.
- Loading is done in **two passes over every file**: pass 1 for global quantile edges +
  session QC, pass 2 for the actual conversion.
- Result: 199 sessions, 38 mice, 51,075 trials, 29,168 neurons.

ii. **Code snippets**

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
all_sessions = get_local_session_metadata(data_root)
print(f"Found {len(all_sessions)} local active experiment files")
...
running_edges, pupil_edges, kept_sessions, image_names = collect_global_statistics(sessions)
...
for idx, session in enumerate(kept_sessions, start=1):
    neural_trials, output_trials, brain_region_idx = convert_session(...)
```

iii. **Justification (CONVERSION_NOTES.md / trajectory)**

- Step 4 discrepancy table: *"Local environment cannot instantiate `NWBFile` for these NWBs due
  `external_resources`/version mismatch … Read NWB files directly with `h5py` and mirror the
  AllenSDK/whitepaper semantics from the processed NWB contents rather than relying on broken
  high-level loading in this environment."*
- Step 5 Key Decision 1: *"Use only locally available active experiment NWBs: The workspace
  contains 284 experiment files, of which 202 are active and 82 passive. Passive sessions collapse
  trial-outcome variability and are outside the active Visual Behavior task."*
- Step 10 Check 3: *"same processed sources are used; only the loader mechanism differs due
  environment incompatibility."*
- Step 6: two-pass design justified as *"pass 1 computes global running-speed and pupil-diameter
  percentile edges and filters out unusable sessions; pass 2 converts trials"*, accepting that
  *"conversion reads each file twice"*.

## 1-b. How are the data split into subjects?

i. **Decisions**

- Subjects are the unique `mouse_id` strings taken from `ophys_experiment_table.csv`, restricted to
  sessions that survive pass-1 QC, and sorted lexicographically.
- `subject_idx` is filled per converted session from a `mouse_id -> index` dict.
- Result: 38 subjects (the reference gets 37; the extra mouse comes from the Multiscope
  experiments the AI also included).

ii. **Code snippets**

```python
SessionMeta(..., mouse_id=str(row.mouse_id), ...)
...
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. **Justification**

- Step 5 variable mapping: *"`mouse_id` from `ophys_experiment_table.csv` → `subjects`,
  `subject_idx`; Unique string list + per-session index."*
- Step 10 Check 4 verified subject count against the raw metadata: *"Included subjects: raw `38`,
  converted `38`."*

## 1-c. How are the data split into sessions?

i. **Decisions**

- **One converted "session" == one `ophys_experiment_id` (one imaging plane)**, not one
  `ophys_session_id`. Planes recorded simultaneously in a Multiscope session are *not* merged; they
  become separate entries in `neural`/`output`, each carrying an identical copy of the same
  behavioural/stimulus timeline.
- Sessions are ordered by `ophys_experiment_id` (no explicit sort by `date_of_acquisition`).
- Passive sessions (`passive == True`) are dropped entirely.
- This is visible in the verifier output: runs of identical trial counts
  (`… 209, 209, 209, 209, 209, 209, 209, … 287 ×7, 309 ×7, 239 ×5, 196 ×5 …`) are the 5–7 planes of
  single Multiscope sessions.

ii. **Code snippets**

```python
exp_table = exp_table[~exp_table["passive"]].copy()
exp_table = exp_table.sort_values("ophys_experiment_id")
for row in exp_table.itertuples(index=False):
    sessions.append(SessionMeta(ophys_experiment_id=int(row.ophys_experiment_id),
                                ophys_session_id=int(row.ophys_session_id), ...))
```

```python
def convert_session(session: SessionMeta, ...):
    with h5py.File(session.filepath, "r") as f:   # one file == one session
        ...
        brain_region_idx = get_cell_count_and_region_idx(f, region_to_idx[session.targeted_structure])
```

iii. **Justification**

- Step 5 Key Decision 2: *"Treat each `ophys_experiment_id` file as one converted session: This
  matches the AllenSDK object granularity (`BehaviorOphysExperiment`) and yields a single imaging
  plane / neuron set / brain region per session."*
- Step 10 Check 5: *"Multiple experiment files can share the same `behavior_session_id` or
  `ophys_session_id`; the conversion intentionally treats each `ophys_experiment_id` plane as a
  separate session because that is the AllenSDK experiment granularity and each file has its own
  neuron set."*
- Step 4: passive sessions excluded because *"all outcomes collapse to `miss`/`correct_reject`"*.

## 1-d. How are the data split into trials?

i. **Decisions**

- Trials come from the processed NWB trial table `intervals/trials` (the Allen `Trials` object
  written to NWB), not from re-derived stimulus logs.
- Kept trials are `(go | catch) & ~aborted & ~auto_rewarded`.
- The trial window is the full `start_time → stop_time` interval (variable length, ~8 s), which
  covers the pre-change flash train and the post-change response window.
- Within that window a uniform 30 Hz grid of bin centres is built
  (`start + (i + 0.5)/30`, keeping centres `< stop_time`), giving mean T = 254 bins.

ii. **Code snippets**

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

iii. **Justification**

- Step 5 Key Decision 3: *"Segment trials using the processed NWB `trials` table: This mirrors the
  Allen reference processing without re-deriving trial logic from lower-level files."*
- Step 5 Key Decision 4: *"Keep only contingent trials: Include GO and CATCH trials; exclude
  `aborted` and `auto_rewarded` exactly as required and consistent with reference definitions."*
- Step 1 notes cite `trial_masks.contingent_trials`, which *"explicitly defines contingent trials as
  GO and CATCH only"*.
- Step 10 Check 5: *"Trial binning uses centers strictly within `[start_time, stop_time)`, avoiding
  off-by-one inclusion past trial end."*

## 1-e. How are trials filtered based on quality controls?

i. **Decisions**

Trial level:
- `(go | catch) & ~aborted & ~auto_rewarded` (no explicit `change_time` validity test — the AI never
  checks `change_time.notna()`; empirically that check is a no-op on this data set).
- Trials whose 30 Hz grid is empty (`centers.size == 0`) are skipped.
- Trials whose neural block is entirely zero (4.84 % of kept trials, 2,474/51,075) are **kept** on
  purpose.

Session level:
- Passive sessions dropped up front.
- Sessions missing the `acquisition/EyeTracking` group dropped (3 sessions: 795953296, 806456687,
  833631914).
- Sessions with fewer than 2 kept trials dropped (pass 1 check; pass 2 raises if it happens there).

Neuron level: nothing (see 2-c).

ii. **Code snippets**

```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
if len(trials) < 2:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: fewer than 2 kept trials")
    continue
```

```python
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
...
except KeyError as exc:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")
```

```python
if len(neural_trials) < 2:
    raise RuntimeError(f"Session {session.ophys_experiment_id} has fewer than 2 usable trials")
```

iii. **Justification**

- Step 5 Key Decision 11: *"Require pupil availability at session level: Three active local files
  lack eye-tracking acquisition entirely; these sessions will be excluded."*
- Step 10: *"All-zero event trials are retained because they are present in the source data and
  still have valid behavioral/stimulus labels."* The AI verified this against the raw NWB
  (experiment 792815735, kept trial 2: raw `event_detection` segment is exactly 0 across all 27
  neurons and 388 native frames).
- Step 10 Check 4: recomputing `(go|catch) & ~aborted & ~auto_rewarded` straight from raw tables
  gave 51,075, *"exactly matching the converted dataset"*.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. **Decisions**

- Neural data is the **detected calcium event magnitude trace**, read from
  `processing/ophys/event_detection/data` (time × ROI) with `…/event_detection/timestamps` as the
  ophys time base.
- dF/F (`processing/ophys/dff`) is explicitly *not* used, and the SDK's smoothed `filtered_events`
  are explicitly *not* used.

ii. **Code snippets**

```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. **Justification**

- Step 4 discrepancy table: *"Paper methods explicitly use detected calcium events for neural
  analyses … Use raw event magnitude traces from `processing/ophys/event_detection/data` as
  `neural`. Do not use `filtered_events` (visualization-only) and do not recompute dF/F."*
- Step 3: *"Paper analyses frequently use discrete calcium events rather than raw dF/F
  (`methods.txt:179`, `methods.txt:208`)"*; *"Whitepaper event detection uses FastLZero on
  fluorescence-derived traces; factor 2.0 at 31 Hz and 2.6 at 11 Hz."*
- Step 5 Key Decision 5: *"Use event traces as neural activity: This best matches the paper's neural
  analyses and uses the whitepaper-defined event detection pipeline already embedded in the NWB
  files."*

## 2-b. How is the `neural` data processed?

i. **Decisions**

- The only processing applied is **linear interpolation of the event magnitude traces from the
  native ophys timestamps onto the per-trial 30 Hz grid of bin centres**. No smoothing, no
  z-scoring, no baseline subtraction, no neuron-wise normalisation.
- The interpolation is implemented with a vectorised `searchsorted` + broadcast gather over all
  neurons at once (rather than a per-neuron `np.interp` loop), and the result is transposed to
  (n_neurons, n_timepoints) `float32`.
- Planes from the same physical session are *not* combined (each plane is its own session), so
  there is no cross-plane stacking step.

ii. **Code snippets**

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

iii. **Justification**

- Step 5 variable mapping: *"Transpose to ROI × time, then linearly interpolate event magnitudes
  from native ophys timestamps onto a common 30 Hz trial grid."*
- Step 6: *"Vectorized linear interpolation for neural event matrices using `searchsorted` +
  broadcasting rather than per-neuron `np.interp`."*
- Step 10 Check 2: the interpolated trial matrices were re-derived independently from raw
  `event_detection` and matched with `np.allclose` (max abs diff 2.98e-08, 0.0, 1.19e-07 on three
  spot-checked trials).

## 2-c. How is the `neural` data filtered based on quality controls?

i. **Decisions**

- **No neuron-level filtering is applied.** Every ROI in
  `processing/ophys/image_segmentation/cell_specimen_table` is kept, and the neuron count is simply
  the number of rows in that table.
- The AI justified this by checking that the released NWBs already contain only valid ROIs
  (29,168 valid of 29,168 listed), i.e. the SDK's `valid_roi` filter is already baked in.

ii. **Code snippets**

```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

iii. **Justification**

- Step 10 Check 3: *"Neuron filtering: reference: `CellSpecimens.__init__` keeps `valid_roi == True`;
  converter: included-session raw NWB files already had all listed cells valid (`29,168` total valid
  of `29,168` total listed), so event matrices matched converted neuron counts exactly."*
- Step 3 records the whitepaper ROI-exclusion rules (motion border, duplicate/union, dendrite,
  too small/dim) but notes they are already applied upstream of the released NWB.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. **Decisions**

- Alignment event is **trial start** (`trials.start_time`); `metadata['temporal_alignment_event'] =
  "trial start"`, `off_start = 0.0`, `off_end = None` (variable-length trials).
- Alignment is done in absolute session time: the trial grid is
  `start_time + (i+0.5)·(1/30)` and every stream (neural, running, pupil, stimulus, outcome) is
  evaluated on **that same grid**, so all streams are aligned by construction.
- Neural values at those times come from interpolating the ophys event trace on its own timestamps,
  so no index-offset arithmetic is involved.

ii. **Code snippets**

```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
if centers.size == 0:
    continue

neural_trial = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
pupil_trial   = linear_resample_vector(pupil_time,  pupil_diameter, centers)
```

```python
"temporal_alignment_event": "trial start",
"off_start": 0.0,
"off_end": None,
```

iii. **Justification**

- Step 5 Key Decision 7: *"Align by absolute ophys time, then cut into trials: For each trial,
  create bin centers from trial `start_time` to `stop_time` at 30 Hz and sample/interpolate all
  streams onto that grid."*
- Step 7 plot review: *"event traces and 30 Hz resampled traces aligned on the same trial window …
  change indicator aligned to the change flash … No obvious temporal misalignment was observed."*
- Step 10 Check 2: independent raw reconstructions of three trials matched with `np.allclose`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Decisions**

- **Yes, temporal rebinning is applied.** Everything is resampled onto a fixed 30 Hz grid,
  `DT = 1/30 s`, `metadata['time_bin_size'] = 33.333… ms`.
- Native rates are ~31 Hz for single-plane `VisualBehavior` experiments (median Δt = 32.3 ms) and
  ~11 Hz per plane for Multiscope; the 30 Hz grid therefore slightly *down*samples single-plane data
  and roughly **3× up**samples Multiscope planes. Rebinning is done by linear interpolation, not by
  summing/averaging events into bins.
- Consequence: mean T = 254 bins/trial (range 211–377); `converted_data.pkl` is 8.1 GB.

ii. **Code snippets**

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

iii. **Justification**

- Step 4 discrepancy table: *"Native sampling rates … Resample all streams to a common 30 Hz grid
  (`33.333... ms` bins). This preserves ophys-based timing while satisfying the decoder requirement
  that all trials/sessions share a common bin size."*
- Step 5 Key Decision 6: *"Use a common 30 Hz trial grid: Native acquisition rates vary across rigs
  (31 Hz single-plane, 11 Hz multiplane). Resampling all streams to 30 Hz gives one shared bin size
  while remaining close to behavior/eye-tracking rate and consistent with paper event-triggered
  interpolation onto 30 Hz timestamps."*

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. **Decisions**

- Image identity is derived from the **stimulus presentation interval table**, not from the trials
  table. The AI scans `intervals/*` for any group whose `stimulus_block_name` contains
  `change_detection` (i.e. `Natural_Images_Lum_Matched_set_*_presentations`) and reads
  `start_time`, `stop_time`, `image_name`, `omitted`, `is_change`, `trials_id`.
- The trials table's `initial_image_name` / `change_image_name` are read but only used as a
  cross-check, never to build the output.

ii. **Code snippets**

```python
def get_task_presentations(f: h5py.File) -> pd.DataFrame:
    for name, group in f["intervals"].items():
        if name == "trials":
            continue
        block_names = decode_str_array(group["stimulus_block_name"][:])
        keep = np.array(["change_detection" in x for x in block_names], dtype=bool)
        ...
        columns = ["start_time", "stop_time", "image_name", "omitted", "is_change",
                   "trials_id", "stimulus_block_name", "active", "duration"]
        df = read_interval_table(group, columns)
        df = df.loc[keep].copy()
```

iii. **Justification**

- Step 4: *"Use stimulus presentation interval tables to build time-varying image identity and
  image-change signals on the resampled trial grid; use trial table to sanity-check change times and
  image names."*
- Step 5 Key Decision 12: *"Restrict stimulus rows to the task block: Use interval groups / block
  labels corresponding to `change_detection_behavior` and trial overlap, ignoring natural-movie or
  spontaneous blocks stored elsewhere in NWB."*

## 3-b. What processing is involved in computing `output` *Image identity*?

i. **Decisions**

- Every bin is initialised to a dedicated **`gray`** category. For each stimulus presentation
  belonging to the current trial (`trials_id == trial['id']`), the bins whose centre falls in
  `[start_time, stop_time)` are set to that flash's image code. Omitted flashes
  (`omitted == True` or `image_name == 'omitted'`) are also set to `gray`.
- So the output alternates image → gray → image → gray at the 250 ms/500 ms stimulus cadence.
- Image names are mapped to integer codes by a **global** mapping collected in pass 1 over all
  sessions, sorted alphabetically with `gray` forced to index 0 → 17 categories
  (`gray` + 16 images).
- Resulting distribution: `gray` 0.670, each of the 16 images ~0.019–0.022. (The reference, by
  contrast, holds the image identity across the grey ISI and has 16 categories at ~6.2 % each.)

ii. **Code snippets**

```python
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
image_change = np.zeros(centers.shape[0], dtype=np.int64)

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

```python
image_values = sorted(image_names)
if "gray" in image_values:
    image_values = ["gray"] + [x for x in image_values if x != "gray"]
image_value_to_idx = {name: idx for idx, name in enumerate(image_values)}
```

iii. **Justification**

- Step 5 Key Decision 9: *"Encode gray/omission periods explicitly: `image_identity` will include a
  `gray` category for ISI and omitted-image periods, because the user specifically asks for the
  image identity during non-gray screen and those periods still occupy trial time."*
- Step 5 Key Decision 10 (analogous global-mapping rationale for running/pupil): codes are built
  globally *"so class semantics are consistent dataset-wide."*
- Step 7 plot review: *"image flashes alternating with gray periods at the expected 250 ms / 500 ms
  cadence."* The 0.670 gray fraction matches the theoretical 500/750 duty cycle.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. **Decisions**

- The image trace is written directly onto `centers`, the **same 30 Hz bin-centre array used for the
  neural matrix**, using half-open interval membership `[start_time, stop_time)` of each flash.
- No separate interpolation or index shift is applied, so alignment is exact by construction.

ii. **Code snippets**

```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
image_identity[mask] = image_value_to_idx[str(row.image_name)]
...
output_trial = np.vstack([image_identity, image_change, running_bin, pupil_bin, outcome_trace])
```

iii. **Justification**

- Step 5 Key Decision 7 (single shared grid for all streams).
- Step 10 Check 2: *"`image_identity`, `image_change`, `running_speed_bin`, `pupil_diameter_bin`,
  `trial_outcome` all matched exactly"* against independent raw reconstruction for three trials.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. **Decisions**

- Derived from the `is_change` boolean column of the same `change_detection` stimulus presentation
  table, restricted to presentations whose `trials_id` equals the current trial id.
- The trials table's `change_time` / `go` columns are read but not used for this output.
- (Empirically equivalent to the reference's `go` gating: `is_change` is True for exactly one flash
  in each go trial and never in a catch trial, which I verified directly on the raw data.)

ii. **Code snippets**

```python
columns = ["start_time", "stop_time", "image_name", "omitted", "is_change", "trials_id", ...]
...
out["is_change"] = out["is_change"].fillna(0).astype(bool)
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. **Justification**

- Step 5 variable mapping: *"Binary per-bin trace: 1 during change-image flash interval, else 0 …
  `is_change`; trial `change_time` for cross-check. Change and pre-change flashes are never
  omitted."*
- Step 1 notes `get_stimulus_presentations` / `is_change_event` as the SDK functions that *"mark
  image identity changes from successive non-omitted images."*

## 4-b. What processing is involved in computing `output` *Image change*?

i. **Decisions**

- A zero vector over the trial's bins, set to 1 **only for the bins covered by the changed flash
  itself** — i.e. a ~250 ms window `[flash start, flash stop)`, ~7–8 bins at 30 Hz.
- The following 500 ms grey ISI is *not* marked (the reference marks a 750 ms window = flash +
  grey).
- Result: 2.5 % of bins are `change` (reference: 7.65 %).

ii. **Code snippets**

```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    ...
    if bool(row.is_change):
        image_change[mask] = 1
```

iii. **Justification**

- Step 5 variable mapping: *"1 during change-image flash interval, else 0."*
- Step 9 consistency table: *"Image change distribution … Flash-change task implies sparse
  positives … [0.975, 0.025]"*, compared to the same value recomputed from raw presentations.
- Step 12 notes the sparsity explicitly: *"image change has sparse positives (`2.5%` of bins) but
  balanced accuracy accounts for class imbalance."*

## 4-c. How is `output` *Image change* thresholded into categories?

i. **Decisions**

- No thresholding of a continuous quantity is needed: the variable is binary by construction,
  0 = `no_change`, 1 = `change`, taken directly from the boolean `is_change` flag.
- `output_values[1] = ["no_change", "change"]`.

ii. **Code snippets**

```python
out["is_change"] = out["is_change"].fillna(0).astype(bool)
...
if bool(row.is_change):
    image_change[mask] = 1
...
"output_values": [image_values, ["no_change", "change"], RUNNING_BIN_NAMES, PUPIL_BIN_NAMES, OUTCOME_NAMES],
```

iii. **Justification**

- Implicit in Step 5's mapping table (binary per-bin trace). The instructions specify image change
  as a binary variable, so no discretisation policy was required.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. **Decisions**

- Identical mechanism to image identity: the same `mask` over the shared `centers` array used for
  the neural matrix, so the change flag is on exactly the same 30 Hz bins as the neural data.

ii. **Code snippets**

```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
if bool(row.is_change):
    image_change[mask] = 1
...
neural_trial = linear_resample_matrix(ophys_time, events, centers)
```

iii. **Justification**

- Step 5 Key Decision 7 (one grid for everything).
- Step 7: *"change indicator aligned to the change flash"* in `processing_*.png`; Step 10 Check 2
  `np.allclose` spot checks.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. **Decisions**

- `processing/running/speed` — the SDK's **filtered** running speed (`data` + `timestamps`), in
  cm/s. The unfiltered variant `processing/running/speed_unfiltered` and the raw `dx` were noted in
  Step 2 but not used.

ii. **Code snippets**

```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. **Justification**

- Step 5 variable mapping: *"`processing/running/speed` (`data`, `timestamps`) → …
  `RunningSpeed.from_stimulus_file`; running-processing module. Use filtered running speed in
  cm/s."*
- Step 1 notes that the SDK *"computes running speed from wheel signals on stimulus timestamps with
  no monitor delay and optional low-pass filtering."*

## 5-b. What processing is involved in computing `output` *Running speed*?

i. **Decisions**

- Linear interpolation (`np.interp`, which clamps rather than extrapolates outside the recorded
  range) from the native ~60 Hz running timestamps onto the trial's 30 Hz bin centres.
- Then discretisation into 5 bins using **global** quantile edges computed in pass 1 from the
  concatenation of all per-trial resampled running traces across all kept sessions.
- No smoothing, no absolute value, no clipping of negative speeds.

ii. **Code snippets**

```python
def linear_resample_vector(src_time, src_value, dst_time) -> np.ndarray:
    ...
    return np.interp(dst_time, src_time, src_value).astype(np.float32, copy=False)
```

```python
running_all = np.concatenate(running_values).astype(np.float64, copy=False)
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
...
running_trial = linear_resample_vector(running_time, running_speed, centers)
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. **Justification**

- Step 5 variable mapping: *"Interpolate filtered running speed onto 30 Hz trial grid; discretize
  globally across included data into 5 equal-frequency bins."*
- Step 5 Key Decision 10: *"Discretize running and pupil globally across the included dataset: Five
  equal-percentile bins will be computed from all finite samples across all included
  sessions/trials, not per session, so class semantics are consistent dataset-wide."*
- Step 10 Check 2: an independent raw re-interpolation of a trial matched the converted
  pre-discretisation trace with `np.allclose`.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. **Decisions**

- `np.quantile` at `[0, .2, .4, .6, .8, 1]` over all finite pooled samples gives 6 edges; ties are
  broken by an epsilon-cumulative-max so the edges are strictly increasing.
- Assignment clips values to `[edge0, edge5]` then uses `np.searchsorted(edges[1:-1], …,
  side="right")`, giving labels 0–4 named `q1…q5`.
- Because the pass-1 pool is exactly the pool that ends up in the file, the realised distribution is
  exactly `{q1 .200, q2 .200, q3 .200, q4 .200, q5 .200}`.

ii. **Code snippets**

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

iii. **Justification**

- Decoder Task spec: *"Running speed, discretized into five equal percentile bins."*
- Step 9: *"running and pupil bins were globally balanced at `~0.2` per class."*
- Step 7 caveat: *"Per-session running and pupil bin distributions are skewed in the 2-session
  sample, which is expected because bin edges are computed globally."*

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. **Decisions**

- Running speed is interpolated directly onto `centers`, the shared 30 Hz trial grid used for the
  neural matrix — the same call signature as the neural resampling, so the two are aligned by
  construction. No lag/monitor-delay correction is applied (the AI notes the SDK already stores
  running speed on delay-free stimulus timestamps).

ii. **Code snippets**

```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial  = linear_resample_matrix(ophys_time,  events,        centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. **Justification**

- Step 1: *"running speed timestamps explicitly require zero monitor delay"*, so both streams live
  on the same hardware-synced clock.
- Step 7 plots: *"running and pupil streams smoothly aligned to the trial grid."*

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. **Decisions**

- `acquisition/EyeTracking/pupil_tracking/{width, height}` with
  `acquisition/EyeTracking/eye_tracking/timestamps` as the time base.
- Diameter is defined as `2 · max(width, height)` — i.e. the major axis of the fitted pupil ellipse
  (width/height are the ellipse semi-axes, since `area == π·width·height`).
- Blinks are handled implicitly: in these NWBs the blink frames are already written as `NaN` in
  `pupil_tracking` (I verified NaN frames coincide exactly with `likely_blink == True`), and the AI
  fills them by time interpolation. The `likely_blink` dataset itself is never read.
- Sessions with no `EyeTracking` group at all are dropped.

ii. **Code snippets**

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

iii. **Justification**

- Step 5 variable mapping: *"Compute pupil diameter as `2 * max(width, height)` after blink
  filtering; interpolate onto 30 Hz grid; discretize globally into 5 equal-frequency bins …
  Exclude sessions with missing eye-tracking acquisition entirely; interpolate within-session for
  blink-related NaNs."*
- Step 5 Key Decision 11: *"Remaining sessions have modest blink-related missingness and can be
  filled by time interpolation before discretization."*

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. **Decisions**

- (1) diameter = `2·max(width, height)`; (2) NaN (blink) fill by linear interpolation on the native
  eye-tracking timestamps; (3) linear interpolation onto the trial's 30 Hz bin centres;
  (4) discretisation into 5 global quantile bins, using exactly the same machinery as running speed.
- No unit conversion to mm, no outlier rejection beyond the NaN fill, no smoothing.

ii. **Code snippets**

```python
pupil_time, pupil_diameter = get_pupil_data(f)
...
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

```python
pupil_all = np.concatenate(pupil_values).astype(np.float64, copy=False)
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
```

iii. **Justification**

- Same as 6-a plus Step 5 Key Decision 10 (global percentile binning for dataset-wide consistent
  class semantics).
- Step 10 Check 2: *"Pupil spot-check: independently compute `2*max(width,height)` from raw
  eye-tracking arrays, interpolate for a selected trial, and verify `np.allclose()` to
  pre-discretized converted pupil trace"* — reported as matching.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. **Decisions**

- Identical to running speed: 5 global equal-frequency bins from `np.quantile` at
  `[0, .2, .4, .6, .8, 1]` over all finite pooled samples; labels `q1…q5` (0–4) assigned by
  `clip` + `searchsorted`.
- Realised distribution is exactly 0.200 per class.

ii. **Code snippets**

```python
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
...
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
...
PUPIL_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
```

iii. **Justification**

- Decoder Task spec: *"Pupil diameter, discretized into five equal percentile bins."*
- Step 9: *"running and pupil bins were globally balanced at `~0.2` per class"*; bin edges are
  recorded in `metadata['pupil_bin_edges']`.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. **Decisions**

- Interpolated onto the same `centers` grid as the neural matrix, in the same loop iteration; no
  extra offset or lag correction.

ii. **Code snippets**

```python
neural_trial = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. **Justification**

- Step 5 Key Decision 7; Step 7 plot review (*"running and pupil streams smoothly aligned to the
  trial grid"*); Step 10 Check 2 `np.allclose` spot check.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. **Decisions**

- The four mutually exclusive boolean columns of `intervals/trials`: `hit`, `miss`, `false_alarm`,
  `correct_reject`, checked in that order.
- If none is true the code **raises** rather than falling back to an "other" class (the reference
  uses an `'other'` fallback). Since aborted/auto-rewarded trials are already removed, exactly one
  flag is always set, so this never fires.

ii. **Code snippets**

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

iii. **Justification**

- Step 1: *"Within `Trial._get_trial_data`, aborted trials force `go = catch = auto_rewarded =
  False`; auto-rewarded trials clear hit/miss/false-alarm/correct-reject labels"* — hence the four
  flags are exhaustive and exclusive on the retained trial set.
- Step 5 variable mapping: *"outcome encoded as constant categorical trace across each trial."*

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. **Decisions**

- Map to a fixed integer code 0–3 in the order `hit, miss, false_alarm, correct_reject` and
  **broadcast that constant across every time bin of the trial**, so the (static) outcome is stored
  as a time-varying row of the `(5, T)` output matrix.
- Resulting distribution: hit 0.303, miss 0.571, false_alarm 0.017, correct_reject 0.108.

ii. **Code snippets**

```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)

output_trial = np.vstack([image_identity, image_change, running_bin, pupil_bin, outcome_trace])
```

iii. **Justification**

- Step 5 Key Decision 8: *"Represent all outputs as time-varying traces: … `trial_outcome` will be
  repeated across bins within a trial as a constant categorical trace to keep one consistent
  `(n_output, T)` format."*
- Step 9 consistency table cross-checks the outcome fractions against the raw trial table:
  *"raw `[0.30703867, 0.56759667, 0.01801273, 0.10735193]`, converted identical."*

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. **Decisions**

- **Missing eye tracking**: `get_pupil_data` raises `KeyError`, which pass 1 catches and logs as a
  skipped session (3 sessions lost). There is no equivalent guard in pass 2 — it relies on pass 1
  having already removed those sessions.
- **Blink NaNs in pupil**: filled by linear interpolation over time (`fill_nan_by_time`); an
  all-NaN session would raise `ValueError` (uncaught outside pass 1's `KeyError` handler).
- **Degenerate trials**: trials whose 30 Hz grid is empty (`stop_time <= start_time`, non-finite
  times) are skipped.
- **Short sessions**: <2 kept trials → skipped in pass 1; in pass 2 the same condition raises
  `RuntimeError`, which is *not* caught and would abort the whole conversion.
- **Out-of-range interpolation**: `np.interp` clamps at the stream boundaries;
  `linear_resample_matrix` clips indices to `[1, len-1]` and would *linearly extrapolate* neural
  values past the end of the ophys recording (I checked the raw data: no kept trial actually falls
  outside the ophys timestamp range, so this path is never exercised).
- **All-zero neural trials**: deliberately retained (2,474 trials, 4.84 %), after verifying against
  the raw NWB that the source `event_detection` block really is all zeros.
- **Duplicate quantile edges**: handled by an epsilon-cumulative-max so `searchsorted` stays valid.
- **Unexpected outcome flags**: raise rather than being coerced to an "other" class.

ii. **Code snippets**

```python
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
...
except KeyError as exc:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")
```

```python
if not np.isfinite(start_time) or not np.isfinite(stop_time) or stop_time <= start_time:
    return np.asarray([], dtype=np.float64)
...
if centers.size == 0:
    continue
```

```python
values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
...
if np.unique(edges).size < edges.size:
    eps = np.finfo(np.float64).eps
    edges = np.maximum.accumulate(edges + np.arange(edges.size) * eps)
```

iii. **Justification**

- Step 5 Key Decision 11 (drop sessions without eye tracking; interpolate blink NaNs).
- Step 10 Issues: *"Verifier warnings for zero neural trials: Not fixed by filtering. Direct raw-NWB
  inspection showed that warned trials can be exactly zero in the source `event_detection` matrix
  itself … Resolution: retain these trials because they are valid source-data trials rather than a
  conversion artifact."*
- Step 10 Check 5: *"Trial binning uses centers strictly within `[start_time, stop_time)`, avoiding
  off-by-one inclusion past trial end. Sessions missing `EyeTracking` are excluded up front because
  pupil output is required."*

## 9-a. What are the most time-consuming steps of the code?

i. **Decisions / findings**

- Measured wall-clock from `conversion_full_out.txt`: total 395.6 s for 202 files →
  **pass 1 = 91.7 s** (~0.45 s/session: trials + presentations + running + pupil, plus per-trial
  resampling of running/pupil) and **pass 2 = 295.0 s** (~1.5 s/session).
- The dominant cost is pass 2, and within it the disk read + decompression of the full-session
  `event_detection` matrix (e.g. 140,204 × 13 … × 666 floats) plus the per-trial
  `linear_resample_matrix` gather over all neurons. This is I/O bound, exactly as in the reference.
- A secondary, algorithmic cost is that `presentations[presentations["trials_id"] == trial_id]`
  re-scans the whole (~4,800-row) presentation DataFrame once per trial, i.e. O(n_trials ×
  n_presentations) per session.
- Pickling the 8.1 GB output is also a non-trivial fixed cost.
- The AI printed per-session timings in both passes, so the bottleneck is directly observable.

ii. **Code snippets**

```python
print(f"[pass2 {idx}/{len(kept_sessions)}] session={session.ophys_experiment_id} "
      f"trials={len(neural_trials)} neurons={brain_region_idx.shape[0]} "
      f"mean_T={mean_t:.1f} elapsed={time.time() - t0:.2f}s")
```

```python
ophys_time, events = get_neural_data(f)      # reads the whole (T, N) event matrix
...
neural_trial = linear_resample_matrix(ophys_time, events, centers)   # per trial, all neurons
```

iii. **Justification**

- Step 6: *"Full conversion may still be I/O-heavy because each NWB event matrix must be read from
  disk. Global binning requires a first pass over sessions, so conversion reads each file twice."*
- Step 7 estimates (*"Pass 1 ~0.36 s/session … Pass 2 ~1.86 s/session … ~7.5 min total"*) were close
  to the realised 6.6 min, comfortably under the 15-minute budget, so no further optimisation was
  pursued.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. **Decisions / findings**

- Already vectorised by the AI: the neural interpolation (`linear_resample_matrix` replaces a
  per-neuron `np.interp` loop with one `searchsorted` + broadcast gather).
- Still scalar and vectorisable:
  - `for trial_idx, trial in trials.iterrows()` — the whole per-trial loop; all trials' bin
    grids/slices could be built with one `searchsorted`.
  - `for row in trial_presentations.itertuples()` — the per-flash loop writing `image_identity` /
    `image_change`; this could be a single `np.searchsorted` of `centers` into the flash boundary
    array.
  - the repeated boolean mask `presentations["trials_id"] == int(trial["id"])`, which could be a
    one-time `groupby`/`searchsorted` instead of a per-trial full scan.
  - `decode_str_array` decodes byte strings in a Python `for` loop, run over every presentation
    table in both passes.
- None of these were flagged in CONVERSION_NOTES; the AI stopped optimising once the estimate was
  under budget.

ii. **Code snippets**

```python
for trial_idx, trial in trials.iterrows():
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

iii. **Justification**

- Step 6 "Code speedups added": *"Vectorized linear interpolation for neural event matrices using
  `searchsorted` + broadcasting rather than per-neuron `np.interp`."*
- Step 7: the AI justified stopping there because the projected total (~7.5 min) was under the
  15-minute threshold in the instructions.

## 9-c. What processing does the code repeat multiple times?

i. **Decisions / findings**

- **Every NWB file is opened and read twice** — once in `collect_global_statistics` (pass 1) and
  once in `convert_session` (pass 2). The trial table, the stimulus presentation table, the running
  stream and the pupil stream are all parsed twice.
- **The per-trial running and pupil resampling is computed twice**: pass 1 builds
  `build_trial_bins` + `linear_resample_vector` for every trial purely to accumulate samples for the
  quantile edges, throws the arrays away, and pass 2 recomputes the identical arrays.
- `build_trial_bins` itself is therefore also executed twice per trial.
- The reference avoids this entirely by keeping the extracted per-trial arrays in memory from its
  single pass.

ii. **Code snippets**

Pass 1:
```python
for trial in trials.itertuples(index=False):
    centers = build_trial_bins(float(trial.start_time), float(trial.stop_time))
    running_trial = linear_resample_vector(running_time, running_speed, centers)
    pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
    running_values.append(running_trial)
    pupil_values.append(pupil_trial)
```

Pass 2 (same computation again):
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
running_trial = linear_resample_vector(running_time, running_speed, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. **Justification**

- Step 6: *"Global binning requires a first pass over sessions, so conversion reads each file
  twice"* — acknowledged as a known cost, accepted because *"Session-level streaming design avoids
  storing continuous raw traces for the whole dataset in memory"* (the two-pass design bounds peak
  memory, which matters given the 8.1 GB output).
- The duplicated *running/pupil resampling* specifically is not called out anywhere in the notes.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. **Decisions / findings**

- Pass 1's per-trial resampled running/pupil arrays (~51k trials × 2 streams) are fully materialised
  and then discarded once the 6 quantile edges are computed; raw in-window samples would have
  sufficed, and the same arrays are recomputed in pass 2 (see 9-c).
- `read_interval_table` pulls the `active` and `duration` presentation columns, which are never
  used; `stimulus_block_name` is read twice (once raw for the mask, once through
  `read_interval_table`) and then overwritten.
- `image_names` is accumulated from *all* pass-1 sessions, including the 3 that are then dropped for
  missing eye tracking.
- `brain_regions` / `subjects` are built from the full `kept_sessions` list *before* `--sample`
  truncation, so in sample mode the saved label lists can contain entries no session uses.
- Upsampling ~11 Hz Multiscope planes onto a 30 Hz grid manufactures ~3× more samples than the
  acquisition contains; those interpolated bins carry no new information but are stored and fed to
  the decoder (contributing to the 8.1 GB file).
- The `--show-processing` path re-slices a padded raw event window (`raw_window`) used only for
  plotting.
- `input` is stored as `(0, T)` arrays per trial — required by the format, but the per-trial `T`
  is irrelevant since the arrays are empty.
- The AI's notes do not identify any of these; Step 6 lists only the two-pass re-read as a known
  inefficiency.

ii. **Code snippets**

```python
running_values.append(running_trial)   # pass 1 only; discarded after compute_quantile_edges
pupil_values.append(pupil_trial)
...
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
```

```python
columns = ["start_time", "stop_time", "image_name", "omitted", "is_change",
           "trials_id", "stimulus_block_name", "active", "duration"]   # 'active','duration' unused
```

```python
subjects = sorted({session.mouse_id for session in kept_sessions})
brain_regions = sorted({session.targeted_structure for session in kept_sessions})
if args.sample:
    kept_sessions = kept_sessions[:2]     # truncation happens after the label lists are built
```

iii. **Justification**

- Not addressed in CONVERSION_NOTES beyond Step 6's acknowledgement of the double file read; the AI
  considered the pipeline fast enough (395 s) that it did not look for further waste.
