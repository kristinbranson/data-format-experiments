# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK high-level loaders. It reads the local NWB (HDF5) files directly with `h5py`. Discovery is done from the local project-metadata CSV `project_metadata/ophys_experiment_table.csv`, intersected with the set of `behavior_ophys_experiment_<id>.nwb` files actually present on disk (284 files). It then drops every experiment flagged `passive == True`, leaving 202 "active" experiment files, and sorts them by `ophys_experiment_id`. No filter on `project_code` is applied, so both `VisualBehavior` (single-plane, ~31 Hz) and `VisualBehaviorMultiscope` (multi-plane, ~11 Hz) files are included. Each retained file is then opened twice: once in "pass 1" (to collect global running/pupil quantile statistics and to reject unusable sessions) and once in "pass 2" (to produce the trial matrices). Within a file, the AI reads the processed NWB groups: `intervals/trials`, `intervals/*_presentations`, `processing/ophys/event_detection`, `processing/running/speed`, `acquisition/EyeTracking`, and `processing/ophys/image_segmentation/cell_specimen_table`.

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
with h5py.File(session.filepath, "r") as f:
    trials = get_trial_table(f)
    ...
    presentations = get_task_presentations(f)
    ophys_time, events = get_neural_data(f)
    running_time, running_speed = get_running_data(f)
    pupil_time, pupil_diameter = get_pupil_data(f)
    brain_region_idx = get_cell_count_and_region_idx(f, region_to_idx[session.targeted_structure])
```

iii. From CONVERSION_NOTES.md Step 4/Step 6: *"Local environment cannot instantiate `NWBFile` for these NWBs due `external_resources`/version mismatch … Read NWB files directly with `h5py` and mirror the AllenSDK/whitepaper semantics from the processed NWB contents rather than relying on broken high-level loading in this environment."* The AI also argues the NWB already contains the *processed* Allen tables, so reading them directly is equivalent to what the SDK returns. Passive exclusion is justified in Step 4/Step 5 Key Decision 1: *"Passive sessions collapse trial-outcome variability and are outside the active Visual Behavior task."* The two-pass design is justified because the running/pupil percentile bin edges must be global across the dataset. Note: the trajectory (step 101/102) shows the SDK load was only attempted with `PYTHONPATH=/app/code` prepended, which shadows the installed `allensdk`; the installed SDK (`BehaviorOphysExperiment.from_nwb_path`) does in fact load these files in this environment. The AI never re-tested without the shadowing path.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` strings taken from `ophys_experiment_table.csv` for the *kept* sessions, sorted lexicographically. `subject_idx[i]` is the index of the mouse that owns kept session `i`. The final dataset has 38 subjects.

ii.
```python
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```
```python
mouse_id=str(row.mouse_id),   # in SessionMeta, from ophys_experiment_table.csv
```

iii. CONVERSION_NOTES Step 5 variable-mapping table: *"`mouse_id` from `ophys_experiment_table.csv` → `subjects`, `subject_idx`: Unique string list + per-session index."* The AI treats `mouse_id` as the canonical animal identifier from the Allen metadata, and checks the resulting subject count against the raw table in Step 10 Check 4 (raw 38, converted 38).

## 1-c. How are the data split into sessions?

i. Each **`ophys_experiment_id` NWB file (i.e. each imaging plane) is treated as one "session"** in the output. Planes recorded simultaneously in the same `ophys_session_id` are *not* merged. Because the AI also includes `VisualBehaviorMultiscope` files, 34 of its 199 output sessions come from only 6 real behavioral sessions of a single mouse (`457841`, which therefore appears with "34 sessions" in the verification log). The remaining 165 output sessions are single-plane `VisualBehavior` experiments (1 plane = 1 session). Sessions are ordered by `ophys_experiment_id`. Sessions are dropped if they have <2 contingent trials or if the NWB has no `acquisition/EyeTracking` group (3 such sessions). Result: 202 active files → 199 output sessions.

ii.
```python
exp_table = exp_table[~exp_table["passive"]].copy()
exp_table = exp_table.sort_values("ophys_experiment_id")
for row in exp_table.itertuples(index=False):
    sessions.append(SessionMeta(ophys_experiment_id=int(row.ophys_experiment_id),
                                ophys_session_id=int(row.ophys_session_id), ...))
```
```python
# pass 1 session-level rejection
if len(trials) < 2:
    print(f"[pass1 ...] skip {session.ophys_experiment_id}: fewer than 2 kept trials")
    continue
...
except KeyError as exc:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")
```
```python
brain_region_idx = get_cell_count_and_region_idx(f, region_to_idx[session.targeted_structure])
# one targeted_structure per "session", i.e. per plane
```

iii. CONVERSION_NOTES Step 5 Key Decision 2: *"Treat each `ophys_experiment_id` file as one converted session: This matches the AllenSDK object granularity (`BehaviorOphysExperiment`) and yields a single imaging plane / neuron set / brain region per session."* Step 10 Check 5 acknowledges the consequence explicitly: *"Multiple experiment files can share the same `behavior_session_id` or `ophys_session_id`; the conversion intentionally treats each `ophys_experiment_id` plane as a separate session because that is the AllenSDK experiment granularity and each file has its own neuron set."* Exclusion of eye-tracking-less sessions is Key Decision 11: *"Three active local files lack eye-tracking acquisition entirely; these sessions will be excluded"* (pupil is a required output).

## 1-d. How are the data split into trials?

i. Trials come from the pre-computed Allen trials table stored in the NWB at `intervals/trials`. The AI keeps contingent trials only — `(go | catch) & ~aborted & ~auto_rewarded` — sorts by trial `id`, and uses the full `start_time`→`stop_time` window (mean ≈ 8.5 s, ≈ 254 bins at 30 Hz). A trial is skipped if the window is empty/non-finite. No fixed window around `change_time` is used.

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

iii. CONVERSION_NOTES Step 4/Step 5 Key Decisions 3–4: *"Use the NWB `trials` table directly as the authoritative processed trial definition, then filter to GO/CATCH and exclude aborted/auto-rewarded. This matches Allen processing while avoiding fragile reimplementation."* The AI grounds the GO/CATCH definition in the SDK's own `trial_masks.contingent_trials` (Step 1 table: *"Explicitly defines contingent trials as GO and CATCH only"*) and in the task instruction to include Go and Catch but exclude Aborted and Auto-rewarded. The full `start_time`→`stop_time` window is used so the pre-change flashes and the post-change response window are both inside the trial, which is what makes the time-varying image-identity/image-change outputs meaningful.

## 1-e. How are trials filtered based on quality controls?

i. Trial-level: keep only `(go | catch) & ~aborted & ~auto_rewarded`; skip a trial whose `start_time`/`stop_time` are non-finite or produce an empty bin grid. Session-level: skip a session with <2 contingent trials in pass 1, skip a session with no `EyeTracking` acquisition, and raise (aborting the run) if <2 usable trials survive in pass 2. No engagement/reward-rate filter, no filter on `change_time` validity, and trials whose neural events are entirely zero are deliberately kept. Final: 51,075 trials over 199 sessions.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
if len(trials) < 2:
    ...continue
```
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
if centers.size == 0:
    continue
...
if len(neural_trials) < 2:
    raise RuntimeError(f"Session {session.ophys_experiment_id} has fewer than 2 usable trials")
```

iii. CONVERSION_NOTES Step 3 "Trial curation rules": *"Aborted trials are excluded from rolling performance metrics and should be excluded for this decoder task … Auto/free-reward trials … should be excluded … GO and CATCH trials are the contingent trial types of interest."* The ≥2-trial rule is required by the target-format spec ("at least two trials within each session"). All-zero-event trials were investigated in Step 10 and kept: *"Direct raw-NWB inspection showed that warned trials can be exactly zero in the source `event_detection` matrix itself … retain these trials because they are valid source-data trials rather than a conversion artifact."*

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the **detected calcium events**, `processing/ophys/event_detection/data` (a `time × ROI` matrix) with `processing/ophys/event_detection/timestamps` as the ophys timebase. dF/F (`processing/ophys/dff`) is read by the exploration scripts but deliberately **not** used for `neural`. The SDK's `filtered_events` (causally smoothed) is also not used.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```

iii. CONVERSION_NOTES Step 4 discrepancy table: *"Paper methods explicitly use detected calcium events for neural analyses → Use raw event magnitude traces from `processing/ophys/event_detection/data` as `neural`. Do not use `filtered_events` (visualization-only) and do not recompute dF/F."* Step 3 cites `methods.txt:179`/`methods.txt:208` (*"Paper analyses frequently use discrete calcium events rather than raw dF/F"*) and the whitepaper's FastLZero event-detection description. Step 5 mapping row: *"Use raw event magnitudes, not `filtered_events`."*

## 2-b. How is the `neural` data processed?

i. The event matrix is transposed to `(n_neurons, n_timepoints)` and **linearly interpolated from the native ophys timestamps onto the per-trial 30 Hz bin-centre grid**. No smoothing, normalisation, z-scoring, baseline subtraction or dF/F computation is applied, and no merging of planes occurs (each plane is its own session). The result is cast to `float32`. The interpolation is vectorised over all neurons at once using `searchsorted` + broadcasting.

ii.
```python
def linear_resample_matrix(src_time, src_value, dst_time):
    """Resample a time x features matrix onto dst_time."""
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
...
neural_trials.append(neural_trial.astype(np.float32, copy=False))
```

iii. CONVERSION_NOTES Step 6: *"Vectorized linear interpolation for neural event matrices using `searchsorted` + broadcasting rather than per-neuron `np.interp`."* Step 5 Key Decision 6 motivates the resampling itself (mixed native rates across rigs, need a single bin size). The AI explicitly avoids any further processing on the grounds that the Allen pipeline (motion correction, neuropil subtraction, demixing, event detection) has already been applied upstream.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering. Every ROI in `processing/ophys/image_segmentation/cell_specimen_table` is kept, and the neuron count is taken from that table (it matches the number of columns in the event matrix). The AI verified against the raw files that the released NWBs contain only valid ROIs, so the SDK's `valid_roi` filter is a no-op here (29,168 valid of 29,168 listed).

ii.
```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

iii. CONVERSION_NOTES Step 10 Check 3: *"reference: `CellSpecimens.__init__` keeps `valid_roi == True`; converter: included-session raw NWB files already had all listed cells valid (`29,168` total valid of `29,168` total listed), so event matrices matched converted neuron counts exactly."* Step 3 notes that the upstream Allen pipeline already removed motion-border, duplicate, union, dendritic and dim ROIs before release.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start**. For each trial, bin centres are generated at `start_time + (k + 0.5)·(1/30 s)` for `k = 0,1,…` while the centre stays below `stop_time`; every stream (neural, running, pupil, stimulus) is evaluated on that same absolute-time grid, so all streams share identical indices. `metadata['temporal_alignment_event'] = 'trial start'`, `off_start = 0.0`, `off_end = None` (variable-length trials). Trial length therefore varies (211–377 bins, mean 254).

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial   = linear_resample_matrix(ophys_time, events, centers)
running_trial  = linear_resample_vector(running_time, running_speed, centers)
pupil_trial    = linear_resample_vector(pupil_time, pupil_diameter, centers)
...
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
```
```python
"temporal_alignment_event": "trial start",
"off_start": 0.0,
"off_end": None,
```

iii. CONVERSION_NOTES Step 5 Key Decision 7: *"Align by absolute ophys time, then cut into trials: For each trial, create bin centers from trial `start_time` to `stop_time` at 30 Hz and sample/interpolate all streams onto that grid."* The instruction requires temporal alignment "based on ophys timestamp", which the AI satisfies by using the event-detection timestamps as the neural reference axis and resampling everything onto a grid defined in that same absolute clock. Step 10 Check 2 verified alignment numerically: three trials rebuilt independently from raw NWB matched with `np.allclose` (max abs diff ≤ 1.19e-07), and Step 7 verified it visually in `processing_*.png`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. **Yes — everything is rebinned/resampled to a fixed 30 Hz grid**, i.e. `time_bin_size = 33.333… ms`, identical for every trial and session. The native rates are ~31 Hz (single-plane `VisualBehavior`) and ~11 Hz (multiplane `VisualBehaviorMultiscope`); the neural, running and pupil streams are all brought to the common grid by linear interpolation (so multiplane data is up-sampled ~2.7×, single-plane data is very slightly down-sampled). Stimulus variables are evaluated (not interpolated) at the bin centres.

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

iii. CONVERSION_NOTES Step 4/Step 5 Key Decision 6: *"Native acquisition rates vary across rigs (31 Hz single-plane, 11 Hz multiplane). Resampling all streams to 30 Hz gives one shared bin size while remaining close to behavior/eye-tracking rate and consistent with paper event-triggered interpolation onto 30 Hz timestamps."* This is driven by the target-format requirement that *"Time bins should be the same size for all trials and sessions"*, which cannot hold across the mixed 31 Hz / 11 Hz rigs the AI chose to include.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the **stimulus-presentations interval table** in the NWB — the group whose `stimulus_block_name` contains `change_detection` (e.g. `intervals/Natural_Images_Lum_Matched_set_training_2017_presentations`). The columns used are `image_name`, `omitted`, `start_time`, `stop_time` and `trials_id`; only presentation rows whose `trials_id` equals the current trial's `id` are used. The trials table's `initial_image_name`/`change_image_name` are read but only used for cross-checking/plots, not to build the output.

ii.
```python
def get_task_presentations(f: h5py.File) -> pd.DataFrame:
    for name, group in f["intervals"].items():
        if name == "trials":  continue
        block_names = decode_str_array(group["stimulus_block_name"][:])
        keep = np.array(["change_detection" in x for x in block_names], dtype=bool)
        ...
        columns = ["start_time","stop_time","image_name","omitted","is_change",
                   "trials_id","stimulus_block_name","active","duration"]
```
```python
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
```

iii. CONVERSION_NOTES Step 4/Step 5: *"Use stimulus presentation interval tables to build time-varying image identity and image-change signals on the resampled trial grid; use trial table to sanity-check change times and image names."* Key Decision 12: *"Restrict stimulus rows to the task block … ignoring natural-movie or spontaneous blocks stored elsewhere in NWB."* The AI's rationale is that the flash table is the ground truth for what was actually on the monitor at each moment, whereas the trial table only gives the initial/change image identities.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A per-bin categorical trace. The trace is **initialised to a `gray` category** and then, for each task flash belonging to the trial, the bins whose centre falls in `[flash.start_time, flash.stop_time)` are set to that flash's image code; omitted flashes are written back as `gray`. Image codes come from a **global** vocabulary built in pass 1 from all unique non-omitted `image_name` values across all kept sessions, sorted alphabetically with `gray` forced to index 0 — 17 categories total (`gray` + 16 images). Consequence: because the flash cadence is 250 ms on / 500 ms gray, **67 % of all output bins are labelled `gray`** and each of the 16 real images occupies ~2 % of bins.

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
    if not mask.any():
        continue
    if bool(row.omitted) or str(row.image_name) == "omitted":
        image_identity[mask] = image_value_to_idx["gray"]
    else:
        image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. CONVERSION_NOTES Step 5 Key Decision 9: *"Encode gray/omission periods explicitly: `image_identity` will include a `gray` category for ISI and omitted-image periods, because the user specifically asks for the image identity during non-gray screen and those periods still occupy trial time."* The global, deterministic (sorted) vocabulary is justified so that codes are consistent across sessions and image sets (images A and B sets both appear).

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is written directly onto the same `centers` array used to resample the neural data, using a half-open `[start_time, stop_time)` interval test on the bin centres. So image identity and neural activity share exactly the same time index by construction; there is no separate interpolation or shifting.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
...
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
image_identity[mask] = image_value_to_idx[str(row.image_name)]
```

iii. CONVERSION_NOTES Step 5 Key Decision 7 (single absolute-time grid for all streams) and Step 10 Check 2: the image-identity trace for three spot-checked trials was independently rebuilt from the raw interval tables and *"matched exactly"*. Step 7 notes the diagnostic plots show *"image flashes alternating with gray periods at the expected 250 ms / 500 ms cadence"* and *"change indicator aligned to the change flash"*.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the `is_change` boolean column of the same task stimulus-presentations table, restricted to flashes whose `trials_id` matches the trial. (The trials table's `change_time`/`go` are read but used only for plotting and cross-checks.)

ii.
```python
out["is_change"] = out["is_change"].fillna(0).astype(bool)
...
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])]
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
    ...
    if bool(row.is_change):
        image_change[mask] = 1
```

iii. Step 1 identifies the SDK's `is_change_event` as the canonical routine that *"marks image identity changes from successive non-omitted images"*; Step 5 maps *"Same stimulus-presentation rows → `output[image_change]`: Binary per-bin trace: 1 during change-image flash interval, else 0"*, with the note *"Change and pre-change flashes are never omitted"* (from `methods.txt`). Using `is_change` rather than the trials table automatically restricts the positive class to real changes — catch (sham-change) trials contain no `is_change` flash.

## 4-b. What processing is involved in computing `output` *Image change*?

i. Essentially none beyond the interval test: a zero vector of length `n_bins` is created and set to 1 on the bins covered by the change flash. No smoothing, no extension past the flash, no per-trial special-casing.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
if bool(row.is_change):
    image_change[mask] = 1
```

iii. See 4-a. The AI's Step 9 consistency table records the resulting distribution as `[0.975, 0.025]` and reasons that *"Flash-change task implies sparse positives"*, i.e. one 250 ms flash inside an ~8.5 s trial.

## 4-c. How is `output` *Image change* thresholded into categories?

i. The variable is natively binary, so no thresholding of a continuous quantity is needed. The categorisation is purely temporal: the positive class occupies exactly the **250 ms duration of the change flash** (`start_time`→`stop_time` of that presentation, ≈ 7–8 bins at 30 Hz); everything else, including the 500 ms gray period after the change, is 0. `output_values` = `["no_change", "change"]`. The positive class is 2.5 % of all bins.

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
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
if bool(row.is_change):
    image_change[mask] = 1
```

iii. Step 5 mapping: *"Binary per-bin trace: 1 during change-image flash interval, else 0"*, which the AI ties to the instruction *"Have value of 1 right after a change in image identity, otherwise 0."* In Step 12 the AI notes the sparsity (*"image change has sparse positives (`2.5%` of bins) but balanced accuracy accounts for class imbalance"*) and concludes no change was needed.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity — written on the shared `centers` grid by a half-open interval test on the change flash's own start/stop times, so it is sample-for-sample aligned with the neural matrix.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
if bool(row.is_change):
    image_change[mask] = 1
...
output_trial = np.vstack([image_identity, image_change, running_bin, pupil_bin, outcome_trace])
```

iii. Same rationale as 3-c (single absolute-time grid). Step 10 Check 2 confirmed the `image_change` trace matched an independent raw-NWB reconstruction exactly for three spot-checked trials; the `--show-processing` plots overlay the change indicator on the flash rasters and the trials-table `change_time` line.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `processing/running/speed` in the NWB (`data` = filtered running speed in cm/s, `timestamps` = behavior clock). The unfiltered variant `processing/running/speed_unfiltered` and the raw `dx` are not used.

ii.
```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. Step 5 mapping row: *"`processing/running/speed` (`data`, `timestamps`) → `output[running_speed_bin]` … Use filtered running speed in cm/s"*, referencing the SDK's `RunningSpeed.from_stimulus_file` (Step 1: *"Computes running speed from wheel signals on stimulus timestamps with no monitor delay and optional low-pass filtering"*). This is the same stream the SDK exposes as `dataset.running_speed`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation (`np.interp`, which clamps at the edges rather than producing NaN) from the native ~60 Hz running timestamps onto the trial's 30 Hz bin centres, then discretisation with globally-computed quantile edges. No smoothing, no absolute value, no sign handling (negative speeds are retained and fall into the lowest bin).

ii.
```python
def linear_resample_vector(src_time, src_value, dst_time):
    if dst_time.size == 0:
        return np.asarray([], dtype=np.float32)
    if src_time.size == 0:
        raise ValueError("Cannot resample from an empty source time series")
    return np.interp(dst_time, src_time, src_value).astype(np.float32, copy=False)
...
running_trial = linear_resample_vector(running_time, running_speed, centers)
running_bin = digitize_with_edges(running_trial, running_edges)
```

iii. Step 5 mapping: *"Interpolate filtered running speed onto 30 Hz trial grid; discretize globally across included data into 5 equal-frequency bins."* Step 10 Check 2 confirms: *"Running spot-check: independently interpolate raw running speed timestamps for a selected trial and verify `np.allclose()` to pre-discretized converted running trace"* — matched.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five **equal-percentile (quintile) bins** whose edges are computed once, globally, in pass 1 from the concatenation of the interpolated running trace of **every kept trial of every kept session** (not per session, not from the raw continuous trace). Degenerate (duplicate) edges are nudged apart by machine epsilon. Values are clipped to `[edge0, edgeN]` and assigned with `searchsorted(..., side="right")` on the interior edges. Edges are stored in `metadata['running_bin_edges']`. The resulting global distribution is exactly `[0.200, 0.200, 0.200, 0.200, 0.200]`.

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

iii. Step 5 Key Decision 10: *"Discretize running and pupil globally across the included dataset: Five equal-percentile bins will be computed from all finite samples across all included sessions/trials, not per session, so class semantics are consistent dataset-wide."* This directly implements the instruction *"Running speed, discretized into five equal percentile bins."*

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated onto the same `centers` grid used for the neural matrix, so it is index-for-index aligned. Running speed is never shifted for monitor delay (consistent with the SDK, which applies zero monitor delay to running timestamps).

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial  = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. Step 1 notes *"Stimulus timestamps carry monitor-delay compensation; running speed timestamps explicitly require zero monitor delay"*, and Step 5 Key Decision 7 states all streams are sampled on the same absolute-time trial grid. Verified in Step 10 Check 2 and visually in the `--show-processing` plots (raw vs resampled running overlaid on the trial window).

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. From `acquisition/EyeTracking/pupil_tracking/{width, height}` with `acquisition/EyeTracking/eye_tracking/timestamps`. Diameter is defined as `2 × max(width, height)`, i.e. twice the major semi-axis of the fitted pupil ellipse. Sessions with no `EyeTracking` group raise `KeyError` and are dropped. The `likely_blink` dataset is not read explicitly — the AI relies on the fact that blink frames are already stored as `NaN` in the released NWB (verified: the NaN mask in `pupil_tracking/width` is exactly the `likely_blink` mask).

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

iii. Step 5 mapping row: *"`acquisition/EyeTracking/pupil_tracking/{width,height,timestamps}` plus blink-filtered fields → `output[pupil_diameter_bin]`: Compute pupil diameter as `2 * max(width, height)` after blink filtering … Exclude sessions with missing eye-tracking acquisition entirely; interpolate within-session for blink-related NaNs"*, referencing the SDK's `process_eye_tracking_data` / `filter_on_blinks`. Key Decision 11 justifies the session exclusion by the fact that pupil diameter is a required output.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Three steps: (1) `2 × max(width, height)` per eye-tracking frame; (2) `fill_nan_by_time` — every NaN sample (blinks / failed ellipse fits) is replaced by linear interpolation across the *whole-session* time axis from the surrounding finite samples, raising if the whole session is NaN; (3) linear interpolation of the gap-filled trace onto the trial's 30 Hz bin centres, then quantile discretisation. No unit conversion, no per-session z-scoring, no smoothing.

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

iii. Step 5: *"interpolate within-session for blink-related NaNs"*; Step 1 notes the SDK's `EyeTrackingTable` *"computes blink flags/outlier filtering"*. The AI's rationale is that the NWB already carries the SDK's blink/outlier decision as NaN, so gap-filling by time is the equivalent of the SDK's drop-then-interpolate behaviour. Step 10 Check 2 verified: *"Pupil spot-check: independently compute `2*max(width,height)` from raw eye-tracking arrays, interpolate for a selected trial, and verify `np.allclose()`"* — matched.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Same scheme as running speed: five equal-percentile bins with edges computed once globally in pass 1 over the interpolated pupil trace of every kept trial of every kept session; clip-then-`searchsorted`. Edges stored in `metadata['pupil_bin_edges']`. Resulting distribution is exactly `[0.200 × 5]`.

ii.
```python
pupil_all = np.concatenate(pupil_values).astype(np.float64, copy=False)
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
...
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```
```python
PUPIL_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
```

iii. Step 5 Key Decision 10 (global percentile bins for consistent dataset-wide class semantics), implementing the instruction *"Pupil diameter, discretized into five equal percentile bins."* Step 10 Check 4 / Step 9 confirm the bins are globally balanced at ~0.2 each.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto the same `centers` grid as the neural matrix; identical indices, no shift.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
pupil_trial  = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. Same rationale as running speed (Step 5 Key Decision 7); eye-tracking timestamps in the NWB are already on the hardware-synced session clock. Verified numerically (Step 10 Check 2) and visually (`processing_*.png`, raw/interp vs resampled pupil overlay).

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. From the four mutually exclusive boolean columns of the NWB trials table — `hit`, `miss`, `false_alarm`, `correct_reject` — checked in that fixed priority order and mapped to codes 0–3. If a trial matches none, the code **raises** rather than assigning a fallback label.

ii.
```python
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]

def trial_outcome_index(trial_row: pd.Series) -> int:
    if bool(trial_row["hit"]): return 0
    if bool(trial_row["miss"]): return 1
    if bool(trial_row["false_alarm"]): return 2
    if bool(trial_row["correct_reject"]): return 3
    raise ValueError("Trial has no valid outcome label")
```

iii. Step 1 identifies `Trial._get_trial_data` as the SDK routine that *"Defines `go`, `catch`, `aborted`, `auto_rewarded`, hit/miss/FA/CR logic"* and notes *"auto-rewarded trials clear hit/miss/false-alarm/correct-reject labels"* — which is why auto-rewarded trials must already be excluded for this mapping to be total. Step 5 mapping: *"outcome encoded as constant categorical trace across each trial"*. The AI's planned sanity check *"Outcome consistency: converted `trial_outcome` labels match mutually exclusive raw trial columns"* was reported as passing in Step 10.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The single integer code is broadcast across all time bins of the trial, so the per-trial static label is stored as a constant time-varying row, and stacked as the 5th row of the `(5, n_timepoints)` output matrix. Resulting distribution: hit 0.303, miss 0.571, false_alarm 0.017, correct_reject 0.108.

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

iii. Step 5 Key Decision 8: *"Represent all outputs as time-varying traces … `trial_outcome` will be repeated across bins within a trial as a constant categorical trace to keep one consistent `(n_output, T)` format."* This follows the target-format guidance *"If at all possible, make it time-varying"* while keeping a single uniform output array shape.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Missing eye tracking**: `get_pupil_data` raises `KeyError`, caught in pass 1, session dropped (3 sessions).
- **Blink / failed pupil fits (NaN)**: linearly interpolated across the session time axis (`fill_nan_by_time`); if a whole session is NaN a `ValueError` is raised (not caught, would abort the run).
- **Out-of-range behavioural timestamps**: `np.interp` clamps to the first/last sample instead of producing NaN, so no NaN reaches the discretiser.
- **Degenerate quantile edges** (e.g. a stream that is mostly a constant): edges are made strictly increasing by adding successive machine-epsilons.
- **Bad trial windows**: non-finite or non-positive `start_time`/`stop_time` produce an empty bin grid and the trial is skipped.
- **Too-small sessions**: <2 contingent trials in pass 1 → skipped; <2 usable trials in pass 2 → `RuntimeError` (uncaught).
- **String/byte columns**: decoded defensively; missing columns tolerated by `read_interval_table`; missing trial `id` synthesised with `arange`.
- **All-zero neural trials** (4.84 % of trials, 2,474): kept deliberately after verifying the raw `event_detection` segment really is zero.
- **No outcome flag set**: `ValueError` is raised rather than silently labelling the trial.

ii.
```python
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
...
except KeyError as exc:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")
```
```python
if finite.sum() == 0:
    raise ValueError("All values are NaN")
if finite.all():
    return values
values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
```
```python
if np.unique(edges).size < edges.size:
    eps = np.finfo(np.float64).eps
    edges = np.maximum.accumulate(edges + np.arange(edges.size) * eps)
```
```python
if "id" not in trials:
    trials["id"] = np.arange(len(trials), dtype=np.int64)
```

iii. Step 5 Key Decision 11 and Step 10 Check 5: *"Sessions missing `EyeTracking` are excluded up front because pupil output is required. All-zero event trials are retained because they are present in the source data and still have valid behavioral/stimulus labels. Trial binning uses centers strictly within `[start_time, stop_time)`, avoiding off-by-one inclusion past trial end."* The AI's stated philosophy is to fail loudly on situations it has not seen (missing outcome label, all-NaN pupil) and to exclude rather than impute when a whole required stream is absent.

## 9-a. What are the most time-consuming steps of the code?

i. The run is I/O- and resampling-bound. Measured from `conversion_full_out.txt` over 199 sessions: pass 1 (trials + presentations + running + pupil, plus per-trial running/pupil interpolation) = **91.7 s** total (~0.46 s/session); pass 2 (the same reads plus reading the full `event_detection` matrix and per-trial neural resampling) = **295.0 s** total (~1.48 s/session, up to ~2.5 s for the largest 666-neuron planes); total wall time 395.6 s. The dominant single cost is reading and interpolating the full `(n_frames × n_neurons)` event matrix — the AI notes that the big multiscope planes are the expensive cases. Pickling the 8.6 GB output is also a non-trivial fixed cost.

ii.
```python
events = np.asarray(event_group["data"][:], dtype=np.float32)   # whole session matrix into RAM
...
neural_trial = linear_resample_matrix(ophys_time, events, centers)  # per trial, all neurons
```
```python
print(f"[pass2 {idx}/{len(kept_sessions)}] session={session.ophys_experiment_id} "
      f"trials={len(neural_trials)} neurons={brain_region_idx.shape[0]} "
      f"mean_T={mean_t:.1f} elapsed={time.time() - t0:.2f}s")
```

iii. CONVERSION_NOTES Step 6: *"Full conversion may still be I/O-heavy because each NWB event matrix must be read from disk. Global binning requires a first pass over sessions, so conversion reads each file twice."* Step 7 estimated ~0.36 s/session for pass 1 and ~1.86 s/session for pass 2 (~7.5 min total), and the actual full run (6.6 min) came in under that estimate, so no further optimisation was done. Trajectory step 189: *"Pass 2 is actually coming in faster than the conservative estimate … The larger multiscope planes are the expensive cases."*

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Two remain:
- The **per-trial Python loop** (`for trial_idx, trial in trials.iterrows()`) in both `collect_global_statistics` and `convert_session`. `iterrows()` is the slowest pandas iteration idiom; all the `searchsorted`/slicing work could be done once for all trials of a session. (Note the pass-1 loop already uses the faster `itertuples`.)
- The **inner per-flash loop** `for row in trial_presentations.itertuples()`, which builds a fresh full-length boolean `mask` over `centers` for each of the ~11 flashes in a trial — an O(n_flashes × n_bins) scan that could be a single `np.searchsorted` of the flash boundaries into `centers`.

The neural interpolation loop *was* vectorised (all neurons at once), which the AI called out as its main speed-up.

ii.
```python
for trial_idx, trial in trials.iterrows():          # per-trial, iterrows()
    ...
    for row in trial_presentations.itertuples(index=False):   # per-flash
        mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
        if not mask.any():
            continue
```
```python
# the one loop that was vectorized:
interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
```

iii. CONVERSION_NOTES Step 6 "Code speedups added": *"Vectorized linear interpolation for neural event matrices using `searchsorted` + broadcasting rather than per-neuron `np.interp`."* The AI did not flag the per-trial or per-flash loops, presumably because the Step 7 timing estimate (~7.5 min) was already inside the instructions' 15-minute budget, so no further vectorisation was pursued.

## 9-c. What processing does the code repeat multiple times?

i. Every kept NWB file is **opened and parsed twice**:
- `get_trial_table` and the trial-type filter run once in pass 1 and again in pass 2.
- `get_task_presentations` (which scans every interval group and decodes string arrays) runs twice.
- `get_running_data`, `get_pupil_data` (including the whole-session `2·max(w,h)` and NaN gap-fill) run twice.
- `build_trial_bins` and the running/pupil `np.interp` run twice **for every trial** — once to build the global quantile edges, once to produce the stored bins. The pass-1 interpolated values are then thrown away.

This duplicated work accounts for the whole 91.7 s of pass 1 (~23 % of total runtime).

ii.
```python
# pass 1
trials = get_trial_table(f); presentations = get_task_presentations(f)
running_time, running_speed = get_running_data(f)
pupil_time, pupil_diameter = get_pupil_data(f)
for trial in trials.itertuples(index=False):
    centers = build_trial_bins(float(trial.start_time), float(trial.stop_time))
    running_values.append(linear_resample_vector(running_time, running_speed, centers))
    pupil_values.append(linear_resample_vector(pupil_time, pupil_diameter, centers))
```
```python
# pass 2 — identical reads and identical resampling repeated
trials = get_trial_table(f); presentations = get_task_presentations(f)
running_time, running_speed = get_running_data(f)
pupil_time, pupil_diameter = get_pupil_data(f)
...
running_trial = linear_resample_vector(running_time, running_speed, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. CONVERSION_NOTES Step 6: *"Global binning requires a first pass over sessions, so conversion reads each file twice."* The AI frames this as inherent to computing global percentile edges, and as part of a *"Session-level streaming design [that] avoids storing continuous raw traces for the whole dataset in memory."* It did not cache the pass-1 running/pupil arrays for reuse, even though the (much larger) pass-2 neural arrays are all held in memory anyway.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **All pass-1 interpolated running/pupil traces** are computed per trial and discarded once the 10 quantile-edge numbers are extracted — the edges could have been estimated from the raw (or a subsampled) continuous stream.
- **Unused columns** are read and decoded: `duration`, `active`, `stimulus_block_name` from the presentations table (only used for the block filter), and `catch`, `change_time`, `initial_image_name`, `change_image_name` from the trials table, which never enter the output (only the plot).
- **`fill_nan_by_time` runs over the whole session's pupil trace** (~136 k samples), including the large stretches outside any kept trial and the 5-minute gray-screen periods.
- **`eye_tracking` group is opened** just to read its `timestamps` alongside `pupil_tracking`.
- **Neural up-sampling**: the ~11 Hz multiscope planes are interpolated to 30 Hz, inflating those trials' neural matrices ~2.7× with no added information, which inflates the 8.6 GB output file and decoder memory.
- **The plotting path** re-slices raw event windows (`events[start:stop]`) for the first trial of the first two sessions when `--show-processing` is on.

ii.
```python
running_values.append(running_trial)   # pass 1 only; only used for np.quantile
pupil_values.append(pupil_trial)
...
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
```
```python
columns = ["start_time","stop_time","image_name","omitted","is_change",
           "trials_id","stimulus_block_name","active","duration"]   # active/duration unused
```
```python
diameter = fill_nan_by_time(timestamps, diameter)   # whole-session gap fill
```

iii. The notes do not identify these as waste; the AI's stated position (Step 6/Step 7) is that the design is dominated by NWB I/O and that the measured total (~6.6 min) was comfortably inside the instructions' budget, so it stopped optimising. The one inefficiency it does acknowledge is the double file read (Step 6).
