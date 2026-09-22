# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does **not** use the AllenSDK `VisualBehaviorOphysProjectCache` at all. It reads the local NWB
files directly with `h5py`. The experiment inventory comes from the local metadata CSV
`data/visual-behavior-ophys-1.1.0/project_metadata/ophys_experiment_table.csv`, intersected with the
`behavior_ophys_experiment_<id>.nwb` files actually present on disk (284 files), then filtered to
`passive == False` (202 files) and sorted by `ophys_experiment_id`. **No `project_code` filter is
applied**, so the retained set mixes `VisualBehavior` (165 kept) and `VisualBehaviorMultiscope`
(34 kept) experiments. Every stream is then read from fixed HDF5 paths inside each NWB file:
`processing/ophys/event_detection` (neural), `processing/running/speed`, `acquisition/EyeTracking`,
`intervals/trials`, and the `*_presentations` interval tables. The conversion makes **two passes** over
the whole file set (pass 1 for global discretization edges + session QC, pass 2 for the actual
conversion).

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

iii. From CONVERSION_NOTES.md Step 4: *"Local environment cannot instantiate `NWBFile` for these NWBs
due `external_resources`/version mismatch … Read NWB files directly with `h5py` and mirror the AllenSDK/
whitepaper semantics from the processed NWB contents rather than relying on broken high-level loading in
this environment."* The AI emphasises that the NWB files already contain the **Allen-processed** tables
(trials, stimulus presentations, dF/F, events, blink-filtered eye tracking), so reading them directly
uses the same processed sources as the SDK, only with a different loader. Passive sessions were excluded
because *"Passive sessions are not task performance and make outcome labels degenerate"* (Step 4). The
AI validated this loading path in Step 10 with `np.allclose()` spot checks reconstructing three trials
straight from the raw NWB arrays.


## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` strings taken from `ophys_experiment_table.csv` for the retained
experiments, sorted alphabetically. Each converted session gets one `subject_idx` pointing into that
list. The full run yielded 38 subjects.

ii.
```python
mouse_id=str(row.mouse_id),
...
subjects = sorted({session.mouse_id for session in kept_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx[idx - 1] = subject_to_idx[session.mouse_id]
```

iii. CONVERSION_NOTES.md Step 5 mapping table: *"`mouse_id` from `ophys_experiment_table.csv` → `subjects`,
`subject_idx`; Unique string list + per-session index."* The AI cross-checked in Step 10 Check 4 that the
raw metadata and the converted pickle both give 38 subjects.


## 1-c. How are the data split into sessions?

i. **One converted "session" = one `ophys_experiment_id` (one imaging plane / one NWB file).** The AI
deliberately does *not* group imaging planes by `ophys_session_id`. For the 165 single-plane
`VisualBehavior` experiments this is identical to a recording session, but the 34 retained
`VisualBehaviorMultiscope` planes come from only **6 physical recording sessions in a single mouse**, so
those 6 recordings are emitted as 34 separate "sessions" with identical trial/behaviour/stimulus label
sequences and disjoint neuron subsets. In the verification output this shows up as runs of identical
per-session trial counts (`209 ×7`, `287 ×7`, `309 ×7`, `239 ×5`, `196 ×5`, `265 ×3`) and as one mouse
(457841) contributing 34 of the 199 sessions.

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
```
(there is no `groupby('ophys_session_id')` anywhere in the script; `ophys_session_id` is only recorded
into `metadata['included_ophys_session_ids']`.)

iii. CONVERSION_NOTES.md Step 5, Key Decision 2: *"**Treat each `ophys_experiment_id` file as one
converted session**: This matches the AllenSDK object granularity (`BehaviorOphysExperiment`) and yields
a single imaging plane / neuron set / brain region per session."* And Step 10 Check 5: *"Multiple
experiment files can share the same `behavior_session_id` or `ophys_session_id`; the conversion
intentionally treats each `ophys_experiment_id` plane as a separate session because that is the AllenSDK
experiment granularity and each file has its own neuron set."* The AI noticed the duplication but did not
discuss the consequence of repeating the same behavioural session multiple times.


## 1-d. How are the data split into trials?

i. Trials come from the Allen-processed `intervals/trials` table inside each NWB file. The retained set
is `(go | catch) & ~aborted & ~auto_rewarded`. Each trial spans `start_time → stop_time` (the native,
variable-length trial window, ~7–12 s, mean ≈ 8.5 s), which is then laid out on a fixed 30 Hz grid of
bin centres. 51,075 trials were produced across 199 sessions (mean 256.7/session).

ii.
```python
def get_trial_table(f: h5py.File) -> pd.DataFrame:
    columns = ["id", "start_time", "stop_time", "go", "catch", "aborted", "auto_rewarded",
               "hit", "miss", "false_alarm", "correct_reject", "change_time",
               "initial_image_name", "change_image_name"]
    trials = read_interval_table(f["intervals"]["trials"], columns)
```
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

iii. CONVERSION_NOTES.md Step 4/5: *"Use the NWB `trials` table directly as the authoritative processed
trial definition, then filter to GO/CATCH and exclude aborted/auto-rewarded. This matches Allen processing
while avoiding fragile reimplementation."* The AI backs the GO∪CATCH definition with the reference code:
*"`trial_masks.contingent_trials` confirms the reference definition for kept trial types is GO plus CATCH
only"* (Step 1 notes). Using the full `start_time → stop_time` window rather than a fixed window around
the change keeps the pre-change flashes, which is what makes the time-varying image identity / image
change outputs meaningful.


## 1-e. How are trials filtered based on quality controls?

i. Trial-level: aborted and auto-rewarded trials excluded; trials whose `[start_time, stop_time)` window
yields zero 30 Hz bin centres are skipped. Session-level: sessions with fewer than 2 contingent trials
are dropped in pass 1; sessions whose NWB has **no `acquisition/EyeTracking` group at all** are dropped
entirely (3 sessions: 795953296, 806456687, 833631914); sessions that end up with fewer than 2 usable
trials in pass 2 raise. Trials whose neural (event) trace is entirely zero are **kept** (2,474 trials =
4.84% of the dataset), which is the source of all the warnings in `verification_full_out.txt`.

ii.
```python
trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
if len(trials) < 2:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: fewer than 2 kept trials")
    continue
```
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
if centers.size == 0:
    continue
```
```python
def get_pupil_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
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

iii. From CONVERSION_NOTES.md Step 5 Key Decisions 4 and 11: *"Keep only contingent trials: Include GO
and CATCH trials; exclude `aborted` and `auto_rewarded` exactly as required and consistent with reference
definitions"*, and *"Require pupil availability at session level: Three active local files lack
eye-tracking acquisition entirely; these sessions will be excluded."* For the all-zero trials (Step 10):
*"Direct raw-NWB inspection showed that warned trials can be exactly zero in the source `event_detection`
matrix itself … retain these trials because they are valid source-data trials rather than a conversion
artifact."*


## 2-a. What variables in the raw data is the `neural` data derived from?

i. `processing/ophys/event_detection/data` — the Allen FastLZero **deconvolved calcium event magnitudes**
(time × ROI) with `processing/ophys/event_detection/timestamps` as the ophys timebase. dF/F
(`processing/ophys/dff`) is explicitly read during exploration but **not used**. `filtered_events` are
also explicitly rejected. Neuron identity/count comes from
`processing/ophys/image_segmentation/cell_specimen_table`.

ii.
```python
def get_neural_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    event_group = f["processing"]["ophys"]["event_detection"]
    timestamps = np.asarray(event_group["timestamps"][:], dtype=np.float64)
    events = np.asarray(event_group["data"][:], dtype=np.float32)
    return timestamps, events
```
```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

iii. CONVERSION_NOTES.md Step 4 discrepancy table: *"Paper methods explicitly use detected calcium events
for neural analyses → Use raw event magnitude traces from `processing/ophys/event_detection/data` as
`neural`. Do not use `filtered_events` (visualization-only) and do not recompute dF/F."* The AI cites
`methods.txt:179`, `methods.txt:208` for the paper using events, and the whitepaper's FastLZero
description (factor 2.0 at 31 Hz, 2.6 at 11 Hz) for how events were produced upstream.


## 2-b. How is the `neural` data processed?

i. The only processing is (a) transposing the stored `time × ROI` matrix to `ROI × time` and (b) linear
interpolation of each ROI's event amplitude onto the per-trial 30 Hz bin-centre grid. No normalisation,
smoothing, z-scoring, baseline subtraction, or cross-plane merging is applied; values are cast to
`float32`. The interpolation is vectorised across neurons with `searchsorted` + broadcasting. Note that
outside the ophys timestamp range the index clipping makes this a linear *extrapolation* rather than a
clamp (unlike `np.interp` used for running/pupil).

ii.
```python
def linear_resample_matrix(src_time, src_value, dst_time):
    """Resample a time x features matrix onto dst_time."""
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

iii. CONVERSION_NOTES.md Step 5: *"Transpose to ROI x time, then linearly interpolate event magnitudes
from native ophys timestamps onto a common 30 Hz trial grid … Use raw event magnitudes, not
`filtered_events`."* Step 6: *"Vectorized linear interpolation for neural event matrices using
`searchsorted` + broadcasting rather than per-neuron `np.interp`."* The AI's rationale for not doing more
is that all standard processing (motion correction, neuropil subtraction, demixing, event detection) is
already baked into the released NWB.


## 2-c. How is the `neural` data filtered based on quality controls?

i. No filtering is applied in the conversion script. The AI relies on the Allen pipeline's ROI curation
already present in the released NWB, and verified that every ROI listed in `cell_specimen_table` has
`valid_roi == True` (29,168 of 29,168), so an explicit `valid_roi` filter would be a no-op. All-zero
neurons/trials are retained.

ii. There is no filtering code. The only neuron-related code is the count used for `brain_region_idx`:
```python
def get_cell_count_and_region_idx(f: h5py.File, region_index: int) -> np.ndarray:
    cell_table = f["processing"]["ophys"]["image_segmentation"]["cell_specimen_table"]
    n_cells = len(cell_table["cell_specimen_id"])
    return np.full(n_cells, region_index, dtype=np.int64)
```

iii. CONVERSION_NOTES.md Step 10 Check 3: *"Neuron filtering: reference: `CellSpecimens.__init__` keeps
`valid_roi == True`; converter: included-session raw NWB files already had all listed cells valid
(`29,168` total valid of `29,168` total listed), so event matrices matched converted neuron counts
exactly."* Step 3 documents the whitepaper ROI-rejection rules (motion border, duplicate, union, apical
dendrite, too small/narrow/dim) as already applied upstream.


## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start**. For each trial the script builds bin centres at
`start_time + (k + 0.5)·(1/30)` for `k = 0 …`, keeping only centres strictly inside
`[start_time, stop_time)`, and samples every stream (neural, running, pupil, stimulus) on that same
absolute-time grid. All streams therefore share one index axis by construction. `metadata` records
`temporal_alignment_event = "trial start"`, `off_start = 0.0`, `off_end = None` (trials are
variable-length, so there is no single end offset).

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

iii. CONVERSION_NOTES.md Step 5 Key Decision 7: *"**Align by absolute ophys time, then cut into trials**:
For each trial, create bin centers from trial `start_time` to `stop_time` at 30 Hz and sample/interpolate
all streams onto that grid."* The AI validated alignment numerically (Step 10 Check 2, `np.allclose`
against independent raw reconstructions of 3 trials, max abs diff ≤ 1.2e-7) and visually in
`processing_775614751.png` / `processing_788490510.png`, where the flash/grey 250/500 ms cadence and the
change flash line up with the resampled traces.


## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Fixed **30 Hz**, i.e. `time_bin_size = 33.333 ms`, for every trial and every session. Yes — rebinning
(resampling) is applied: every stream is linearly interpolated onto the common 30 Hz grid. Because
single-plane `VisualBehavior` ophys is acquired at 30.96 Hz, this is essentially a no-op for 165 of the
199 sessions; for the 34 `VisualBehaviorMultiscope` planes (10.73 Hz native) it is a ~2.8× **upsampling**
of sparse deconvolved event traces. Resulting trials are 211–377 bins (mean 254).

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

iii. CONVERSION_NOTES.md Step 4/5 Key Decision 6: *"**Use a common 30 Hz trial grid**: Native acquisition
rates vary across rigs (31 Hz single-plane, 11 Hz multiplane). Resampling all streams to 30 Hz gives one
shared bin size while remaining close to behavior/eye-tracking rate and consistent with paper
event-triggered interpolation onto 30 Hz timestamps."* The driving constraint cited is the target-format
requirement that *"Time bins should be the same size for all trials and sessions."*


## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The per-flash stimulus presentation interval table (`intervals/<image_set>_presentations`), restricted
to rows whose `stimulus_block_name` contains `change_detection`. The columns used are `image_name`,
`omitted`, `start_time`, `stop_time` and `trials_id`. The trials-table fields `initial_image_name` /
`change_image_name` are read but only used for the diagnostic plots, **not** to build the output.

ii.
```python
def get_task_presentations(f: h5py.File) -> pd.DataFrame:
    for name, group in f["intervals"].items():
        if name == "trials":
            continue
        ...
        block_names = decode_str_array(group["stimulus_block_name"][:])
        keep = np.array(["change_detection" in x for x in block_names], dtype=bool)
        ...
        columns = ["start_time", "stop_time", "image_name", "omitted", "is_change",
                   "trials_id", "stimulus_block_name", "active", "duration"]
        df = read_interval_table(group, columns)
        df = df.loc[keep].copy()
```
```python
trial_presentations = presentations[presentations["trials_id"] == int(trial["id"])].copy()
for row in trial_presentations.itertuples(index=False):
    mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
```

iii. CONVERSION_NOTES.md Step 5 mapping table: *"`intervals/*_presentations` task image block
(`image_name`, `omitted`, `is_change`, `start_time`, `stop_time`, `trials_id`, `stimulus_block_name`) →
`output[image_identity]` … Restrict to `stimulus_block_name == change_detection_behavior` when present."*
Key Decision 12: *"Restrict stimulus rows to the task block … ignoring natural-movie or spontaneous
blocks stored elsewhere in NWB."* The AI's argument is that the presentation table is the ground-truth
record of what was on the monitor at each moment, including omissions.


## 3-b. What processing is involved in computing `output` *Image identity*?

i. A per-bin categorical trace is built. It is initialised to a dedicated **`gray`** category and then
overwritten, flash by flash, with the integer code of the presented image for bins whose centre falls in
`[flash.start_time, flash.stop_time)`. Omitted flashes (`omitted == True` or `image_name == "omitted"`)
are mapped back to `gray`. The code book is global across the dataset: the union of image names seen in
pass 1, sorted, with `gray` forced to index 0 — 17 categories (`gray` + 16 natural images from image
sets A and B). Because the stimulus is 250 ms on / 500 ms off, `gray` accounts for 67.0% of all bins
in the converted data (verification output: `gray (0.670)`, each image ≈ 0.019–0.022).

ii.
```python
image_identity = np.full(centers.shape[0], image_value_to_idx["gray"], dtype=np.int64)
...
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

iii. CONVERSION_NOTES.md Step 5 Key Decision 9: *"**Encode gray/omission periods explicitly**:
`image_identity` will include a `gray` category for ISI and omitted-image periods, because the user
specifically asks for the image identity during non-gray screen and those periods still occupy trial
time."* Key Decision 10 explains the global (not per-session) code book so that class semantics are
consistent dataset-wide.


## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is computed directly on the same `centers` array used to resample the neural data, so the alignment
is exact by construction: bin *k* of `neural` and bin *k* of `image_identity` refer to the same absolute
time `start_time + (k + 0.5)/30`. Membership uses a half-open interval `[start, stop)` on the flash
boundaries.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
...
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
...
output_trial = np.vstack([image_identity, image_change, running_bin, pupil_bin, outcome_trace])
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 7 (single shared grid for all streams) plus the Step 10
Check 2 sanity checks, which report *"`image_identity`, `image_change`, `running_speed_bin`,
`pupil_diameter_bin`, `trial_outcome` all matched exactly"* against independent raw-NWB reconstructions
for three trials. The Step 7 plot review also states the *"image flashes alternating with gray periods at
the expected 250 ms / 500 ms cadence"* — and the realised `gray` fraction of 0.670 matches the expected
500/750 exactly, confirming there is no systematic offset.


## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` boolean column of the same task stimulus presentation table (plus that row's
`start_time` / `stop_time`). The trials-table `change_time` and `go` columns are read but used only in the
diagnostic plot / as a cross-check, not to build the output. I verified in the raw data that `is_change`
is True exactly once per go trial and never on catch trials (catch trials carry `is_sham_change` instead),
so this is equivalent to the reference's "go trials only" restriction.

ii.
```python
columns = ["start_time", "stop_time", "image_name", "omitted", "is_change", "trials_id",
           "stimulus_block_name", "active", "duration"]
...
out["is_change"] = out["is_change"].fillna(0).astype(bool)
```
```python
if bool(row.is_change):
    image_change[mask] = 1
```

iii. CONVERSION_NOTES.md Step 5 mapping table: *"Same stimulus-presentation rows → `output[image_change]`;
Binary per-bin trace: 1 during change-image flash interval, else 0; reference fn `is_change_event`; trial
`change_time` for cross-check. Change and pre-change flashes are never omitted."* Step 1 notes cite the
reference `stimulus_processing.is_change_event`, which *"marks image identity changes from successive
non-omitted images"*, as the authority for this column.


## 4-b. What processing is involved in computing `output` *Image change*?

i. A binary per-bin trace, zero-initialised, set to 1 only on the bins covered by the change flash itself,
i.e. a **250 ms window** starting at change onset (~7–8 bins at 30 Hz). It is not extended over the
following grey period and is never set on catch trials. Across the full dataset this gives 2.5% positive
bins (`no_change 0.975 / change 0.025`).

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

iii. From the task spec quoted in CONVERSION_NOTES.md: *"Image change, binary variable. Have value of 1
right after a change in image identity, otherwise 0."* The AI implemented the most literal reading —
exactly the duration of the changed image's presentation. Step 12 records that the resulting sparsity was
checked: *"image change has sparse positives (`2.5%` of bins) but balanced accuracy accounts for class
imbalance."*


## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding — the variable is natively binary (`0 = no_change`, `1 = change`), taken straight from
the boolean `is_change` flag. `output_values[1] = ["no_change", "change"]`.

ii.
```python
image_change = np.zeros(centers.shape[0], dtype=np.int64)
...
if bool(row.is_change):
    image_change[mask] = 1
...
"output_values": [
    image_values,
    ["no_change", "change"],
    RUNNING_BIN_NAMES,
    PUPIL_BIN_NAMES,
    OUTCOME_NAMES,
],
```

iii. Not separately justified beyond the task specification ("Image change, binary variable"); the AI
simply carries the raw boolean through. `verification_full_out.txt` confirms `image_change: [0.0, 1.0]`.


## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity — the same `centers` grid and the same `[flash.start_time,
flash.stop_time)` mask, so the change indicator starts on the first bin whose centre falls inside the
change flash. No lag or shift is applied to compensate for calcium-indicator kinetics.

ii.
```python
mask = (centers >= float(row.start_time)) & (centers < float(row.stop_time))
...
if bool(row.is_change):
    image_change[mask] = 1
...
output_trial = np.vstack([image_identity, image_change, running_bin, pupil_bin, outcome_trace])
```

iii. Same as 3-c: a single shared bin-centre grid for every stream (Step 5 Key Decision 7), validated with
`np.allclose()` raw-data spot checks (Step 10 Check 2) and visually in the processing plots, where the AI
notes the *"change indicator aligned to the change flash"*.


## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed` (`data` + `timestamps`) — the Allen-filtered running speed in cm/s. This is
the same object the SDK exposes as `BehaviorSession.running_speed`; `speed_unfiltered` and `dx` are noted
during exploration but not used.

ii.
```python
def get_running_data(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    running_group = f["processing"]["running"]["speed"]
    timestamps = np.asarray(running_group["timestamps"][:], dtype=np.float64)
    speed = np.asarray(running_group["data"][:], dtype=np.float64)
    return timestamps, speed
```

iii. CONVERSION_NOTES.md Step 5 mapping table: *"`processing/running/speed` (`data`, `timestamps`) →
`output[running_speed_bin]` … reference fn `RunningSpeed.from_stimulus_file`; running-processing module.
Use filtered running speed in cm/s."* Step 1 notes that the reference *"Computes running speed from wheel
signals on stimulus timestamps with no monitor delay and optional low-pass filtering"*, i.e. the stored
`speed` already is the reference product.


## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation (`np.interp`, so constant-clamped outside the recorded range) from the running
timestamps onto the trial's 30 Hz bin centres, then discretisation into 5 bins. Bin edges are computed
**once globally** in pass 1 from the concatenation of all trial-resampled running values over all retained
sessions, using `np.quantile` at `[0, 0.2, 0.4, 0.6, 0.8, 1.0]`, with an epsilon nudge if any edges tie.

ii.
```python
def linear_resample_vector(src_time, src_value, dst_time):
    ...
    return np.interp(dst_time, src_time, src_value).astype(np.float32, copy=False)
```
```python
running_trial = linear_resample_vector(running_time, running_speed, centers)
running_values.append(running_trial)
...
running_all = np.concatenate(running_values).astype(np.float64, copy=False)
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
```
```python
def compute_quantile_edges(values: np.ndarray, nbins: int) -> np.ndarray:
    probs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(values, probs)
    if np.unique(edges).size < edges.size:
        eps = np.finfo(np.float64).eps
        edges = np.maximum.accumulate(edges + np.arange(edges.size) * eps)
    return edges
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 10: *"**Discretize running and pupil globally across the
included dataset**: Five equal-percentile bins will be computed from all finite samples across all
included sessions/trials, not per session, so class semantics are consistent dataset-wide."* The
two-pass structure exists specifically to make this possible. The result is verified in
`verification_full_out.txt`: `running_speed_bin: {q1 0.200, q2 0.200, q3 0.200, q4 0.200, q5 0.200}`.


## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile ("quintile") bins, labelled `q1 … q5` (integer codes 0–4). Values are clipped to
the outer edges then assigned with `searchsorted` on the four interior edges (`edges[1:-1]`, `side="right"`).

ii.
```python
def digitize_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, edges[0], edges[-1])
    bins = np.searchsorted(edges[1:-1], clipped, side="right")
    return bins.astype(np.int64, copy=False)
```
```python
running_bin = digitize_with_edges(running_trial, running_edges)
...
RUNNING_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
```

iii. Directly from the task specification ("Running speed, discretized into five equal percentile bins"),
with the global-edge rationale in Step 5 Key Decision 10. The AI plotted the histogram with edges overlaid
in the `--show-processing` figures and confirmed the resulting 0.200/0.200/0.200/0.200/0.200 marginal.


## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated directly onto the same `centers` bin-centre grid that the neural data is resampled
onto, so the two share one index axis exactly. No monitor-delay or other shift is applied (running
timestamps in the NWB are already sync-derived and carry no monitor delay).

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
running_trial = linear_resample_vector(running_time, running_speed, centers)
```

iii. Step 5 Key Decision 7 (shared grid). Step 1 explicitly notes the sync provenance: *"Stimulus
timestamps carry monitor-delay compensation; running speed timestamps explicitly require zero monitor
delay."* Step 10 Check 2 verified a running trace by independent raw interpolation with `np.allclose()`.


## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/width` and `.../height` together with
`acquisition/EyeTracking/eye_tracking/timestamps`. Diameter is defined as `2 · max(width, height)`.
Blink frames are handled implicitly: the released NWB already has `width`/`height` set to `NaN` wherever
`likely_blink` is True (I confirmed `isnan(width) == likely_blink` exactly, 9.2% of frames in a spot-checked
session), and those NaNs are then filled by time interpolation.

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

iii. CONVERSION_NOTES.md Step 5 mapping table: *"`acquisition/EyeTracking/pupil_tracking/{width,height,
timestamps}` plus blink-filtered fields → `output[pupil_diameter_bin]`; Compute pupil diameter as
`2 * max(width, height)` after blink filtering … reference fns `process_eye_tracking_data`,
`filter_on_blinks`, whitepaper eye-tracking description."* This mirrors the SDK's own
`compute_circular_area`, which uses `max(width, height)` as the pupil radius.


## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) `2 · max(width, height)`; (2) fill blink-induced NaNs by linear interpolation over the eye-tracking
time axis; (3) linear interpolation onto the trial's 30 Hz bin centres; (4) discretisation into 5 global
equal-percentile bins using exactly the same machinery as running speed. Sessions with no eye-tracking
group at all are dropped rather than NaN-filled.

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
pupil_values.append(pupil_trial)
...
pupil_all = np.concatenate(pupil_values).astype(np.float64, copy=False)
pupil_edges = compute_quantile_edges(pupil_all[np.isfinite(pupil_all)], 5)
```

iii. CONVERSION_NOTES.md Step 5 Key Decisions 10 and 11: global 5-quantile discretisation, and
*"Exclude sessions with missing eye-tracking acquisition entirely; interpolate within-session for
blink-related NaNs."* The AI's stated reason for interpolating rather than dropping blink bins is that
blink gaps are short and *"Remaining sessions have modest blink-related missingness and can be filled by
time interpolation before discretization."*


## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five equal-percentile bins (`q1 … q5`, codes 0–4), from globally computed edges, applied with the same
clip + `searchsorted` helper as running speed. Realised marginal is exactly 0.200 per bin.

ii.
```python
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
...
PUPIL_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
...
"pupil_bin_edges": pupil_edges.tolist(),
```

iii. Task specification ("Pupil diameter, discretized into five equal percentile bins") plus Step 5 Key
Decision 10 for the global-edges choice; verified in `verification_full_out.txt`
(`pupil_diameter_bin: {q1 0.200, …, q5 0.200}`).


## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same as running speed — interpolated onto the identical `centers` bin-centre grid used for the neural
matrix, so index *k* of both refers to the same absolute time. The eye-tracking timestamps in the NWB are
already sync-derived onto the same master clock as the ophys timestamps, so no further correction is
applied.

ii.
```python
centers = build_trial_bins(float(trial["start_time"]), float(trial["stop_time"]))
neural_trial = linear_resample_matrix(ophys_time, events, centers)
pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
pupil_bin = digitize_with_edges(pupil_trial, pupil_edges)
```

iii. Step 5 Key Decision 7 (shared grid); Step 10 Check 2 verified pupil against an independent raw
recomputation (`2*max(width,height)` + interpolation) with `np.allclose()` for three trials.


## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of `intervals/trials`: `hit`, `miss`, `false_alarm`,
`correct_reject`, checked in that order and mapped to codes 0–3. If none is set the code raises rather
than emitting a fallback label.

ii.
```python
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]
...
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

iii. CONVERSION_NOTES.md Step 5 mapping table: outcome comes from the trials table and is *"encoded as
constant categorical trace across each trial"*, citing `Trial._get_trial_data` as the reference logic. The
AI's Step 1 notes record that *"aborted trials force `go = catch = auto_rewarded = False`; auto-rewarded
trials clear hit/miss/false-alarm/correct-reject labels"*, which is why these four flags are guaranteed
exhaustive on the retained `(go|catch) & ~aborted & ~auto_rewarded` subset — i.e. the `raise` should be
unreachable.


## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The single integer code is broadcast across all bins of the trial so that the output row has the same
`(n_timepoints,)` shape as the other four outputs and the whole `output` array can be `vstack`ed into
`(5, T)`. `output_values[4] = ["hit", "miss", "false_alarm", "correct_reject"]`. Realised distribution:
hit 0.303, miss 0.571, false_alarm 0.017, correct_reject 0.108.

ii.
```python
outcome_idx = trial_outcome_index(trial)
outcome_trace = np.full(centers.shape[0], outcome_idx, dtype=np.int64)

output_trial = np.vstack(
    [
        image_identity,
        image_change,
        running_bin,
        pupil_bin,
        outcome_trace,
    ]
)
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 8: *"**Represent all outputs as time-varying traces**:
`image_identity`, `image_change`, `running_speed_bin`, and `pupil_diameter_bin` are naturally
time-varying; `trial_outcome` will be repeated across bins within a trial as a constant categorical trace
to keep one consistent `(n_output, T)` format."* This follows the target-format guidance *"If at all
possible, make it time-varying."*


## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **No eye-tracking group in the NWB** → `KeyError` raised by `get_pupil_data`, caught in pass 1, whole
  session skipped (3 sessions).
- **Blink NaNs in pupil** → filled by linear interpolation over the eye-tracking time axis before
  resampling (so no synthetic "bin 0" labels are injected).
- **Degenerate trial window** (`stop_time <= start_time`, non-finite times, or no bin centre inside the
  window) → trial skipped.
- **Fewer than 2 contingent trials in a session** → session skipped in pass 1 (and a hard error in pass 2
  if it happens after binning).
- **Tied quantile edges** → nudged apart by machine epsilon so `searchsorted` still yields 5 distinct bins.
- **Missing optional columns** in an interval table → silently omitted by `read_interval_table`; missing
  `id` in the trials table → synthesised as `arange`.
- **Byte-string vs str** NWB columns → normalised by `decode_str_array`.
- **All-zero event traces** (2,474 trials, 4.84%) → deliberately retained after verifying against the raw
  NWB that the source data itself is zero.
- Not handled: pass 2 has no `try/except`, so any unexpected per-session failure there aborts the whole
  run; pass 1 only catches `KeyError`.

ii.
```python
if "EyeTracking" not in f["acquisition"]:
    raise KeyError("Missing EyeTracking acquisition")
...
except KeyError as exc:
    print(f"[pass1 {idx}/{len(sessions)}] skip {session.ophys_experiment_id}: {exc}")
```
```python
values[~finite] = np.interp(time_axis[~finite], time_axis[finite], values[finite])
```
```python
if not np.isfinite(start_time) or not np.isfinite(stop_time) or stop_time <= start_time:
    return np.asarray([], dtype=np.float64)
```
```python
if np.unique(edges).size < edges.size:
    eps = np.finfo(np.float64).eps
    edges = np.maximum.accumulate(edges + np.arange(edges.size) * eps)
```
```python
for column in columns:
    if column not in group:
        continue
    data[column] = read_dataset(group[column])
```

iii. CONVERSION_NOTES.md Step 5 Key Decision 11 and Step 10 Check 5: *"Sessions missing `EyeTracking` are
excluded up front because pupil output is required"*; *"Trial binning uses centers strictly within
`[start_time, stop_time)`, avoiding off-by-one inclusion past trial end"*; *"All-zero event trials are
retained because they are present in the source data and still have valid behavioral/stimulus labels."*
For the zero trials the AI documented a concrete raw-data check (experiment 792815735, kept trial 2,
27 neurons × 388 native frames, raw event sum exactly 0.0).


## 9-a. What are the most time-consuming steps of the code?

i. Disk I/O: reading each NWB's `event_detection/data` matrix (up to 140,204 × 666 float32), the running
trace (~270k samples) and the eye-tracking arrays. Because the design is two-pass, **every file is opened
and parsed twice**. Measured: pass 1 ≈ 0.36 s/session, pass 2 ≈ 1–3 s/session (scaling with neuron count;
the AI noted *"The larger multiscope planes are the expensive cases"*), total 395.6 s for 199 sessions.
Within pass 2 the dominant per-trial cost is `linear_resample_matrix`, which materialises two
`(T_bins, n_neurons)` gathers per trial.

ii.
```python
t0 = time.time()
...
print(f"[pass2 {idx}/{len(kept_sessions)}] session={session.ophys_experiment_id} "
      f"trials={len(neural_trials)} neurons={brain_region_idx.shape[0]} "
      f"mean_T={mean_t:.1f} elapsed={time.time() - t0:.2f}s")
```
```python
interp = src_value[idx_lo] * (1.0 - w[:, None]) + src_value[idx_hi] * w[:, None]
```

iii. CONVERSION_NOTES.md Step 6: *"Full conversion may still be I/O-heavy because each NWB event matrix
must be read from disk. Global binning requires a first pass over sessions, so conversion reads each file
twice."* Step 7 gives the timing estimate (~7.5 min projected, 6.6 min actual), comfortably under the
15-minute budget in the instructions, which is why no further optimisation was pursued.


## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Already vectorised: the neural interpolation across all neurons (`linear_resample_matrix`), replacing a
per-neuron `np.interp` loop. Still scalar loops that could be vectorised:
- The **per-trial loop** in `convert_session` and in `collect_global_statistics` — all trials' bin centres
  could be built as one concatenated grid with per-trial offsets and resampled in a single call.
- The **per-presentation inner loop** (`for row in trial_presentations.itertuples()`), which is O(flashes
  × T_bins) because it builds a boolean `mask` over the whole trial for each of ~11 flashes. This could be
  a single `np.searchsorted(presentation_edges, centers)` lookup per trial, or a whole-session
  `searchsorted` done once.
- `decode_str_array`'s element-wise Python loop, replaceable by `np.char.decode` / `astype(str)`.
- The `[ "change_detection" in x for x in block_names ]` list comprehension.
- `trials.iterrows()` (row-wise Series construction) is slower than `itertuples()`, which the code uses
  elsewhere.

ii.
```python
for trial_idx, trial in trials.iterrows():
    centers = build_trial_bins(...)
    ...
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

iii. CONVERSION_NOTES.md Step 6 lists only the speed-up that was done (*"Vectorized linear interpolation
for neural event matrices using `searchsorted` + broadcasting rather than per-neuron `np.interp`"*). The
remaining loops were left in place because the measured runtime (~6.6 min) already met the instruction's
15-minute threshold, so the AI never revisited them — this trade-off is implicit rather than argued.


## 9-c. What processing does the code repeat multiple times?

i. The two-pass design repeats a substantial amount of work:
- Every NWB file is **opened and read twice**; `get_trial_table`, `get_task_presentations`,
  `get_running_data` and `get_pupil_data` all run once in pass 1 and again in pass 2.
- The **running and pupil traces are resampled onto the trial grid twice** — once in
  `collect_global_statistics` purely to accumulate values for the quantile edges, and again in
  `convert_session` to produce the stored bins. `build_trial_bins` is likewise called twice per trial.
- `fill_nan_by_time` over the full-session pupil array is recomputed in both passes.
- The trial filter expression `(go|catch) & ~aborted & ~auto_rewarded` is written out twice.

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
# pass 2 - the same four readers and the same two resamples again
with h5py.File(session.filepath, "r") as f:
    trials = get_trial_table(f)
    trials = trials[(trials["go"] | trials["catch"]) & (~trials["aborted"]) & (~trials["auto_rewarded"])].copy()
    presentations = get_task_presentations(f)
    ophys_time, events = get_neural_data(f)
    running_time, running_speed = get_running_data(f)
    pupil_time, pupil_diameter = get_pupil_data(f)
    ...
        running_trial = linear_resample_vector(running_time, running_speed, centers)
        pupil_trial = linear_resample_vector(pupil_time, pupil_diameter, centers)
```

iii. CONVERSION_NOTES.md Step 6 names the cause and the justification: *"Global binning requires a first
pass over sessions, so conversion reads each file twice"*, traded against *"Session-level streaming design
avoids storing continuous raw traces for the whole dataset in memory."* Global percentile edges (Step 5
Key Decision 10) genuinely require seeing all data before any trial can be discretised; the AI chose to
pay the re-read rather than cache ~51k trial-length running/pupil vectors (which is only a few hundred MB
and would have removed the duplication).


## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Work performed whose result never reaches `converted_data.pkl`:
- **The entire pass-1 resampling of running and pupil** — 51k trials' worth of interpolated vectors are
  built and concatenated only to compute 12 quantile edges, then thrown away and recomputed in pass 2.
- **Trials-table columns that are never used in the outputs**: `change_time`, `initial_image_name`,
  `change_image_name` are read and type-converted for every session but only consulted by the optional
  `--show-processing` plot; `catch` is used only inside the filter expression.
- **Presentation columns read but unused**: `stop_time`-adjacent extras `duration`, `active`, and
  `stimulus_block_name` (the latter re-assigned into the frame after already being used for the mask).
- **`out.sort_values("start_time")`** on the presentation table, when lookup is by `trials_id` masking
  rather than by order.
- **`fill_nan_by_time` over the whole session's pupil trace**, when only the samples bracketing trial
  windows are ever needed.
- **Per-trial `np.clip` in `digitize_with_edges`** for running, which can never be out of range for the
  pass-1 sessions since the edges were derived from those exact values.
- Minor: `neural_trial.astype(np.float32, copy=False)` after `linear_resample_matrix` already returned
  float32; `SessionMeta` carries `session_type`, `experience_level`, `project_code` which are never read.

ii.
```python
running_values.append(running_trial)   # pass 1: used only for compute_quantile_edges, then discarded
pupil_values.append(pupil_trial)
...
running_edges = compute_quantile_edges(running_all[np.isfinite(running_all)], 5)
```
```python
columns = ["id", "start_time", "stop_time", "go", "catch", "aborted", "auto_rewarded",
           "hit", "miss", "false_alarm", "correct_reject", "change_time",
           "initial_image_name", "change_image_name"]   # last three unused outside plotting
...
for column in ["initial_image_name", "change_image_name"]:
    if column in trials:
        trials[column] = trials[column].astype(str)
```
```python
df["stimulus_block_name"] = block_names[keep]
...
return out.sort_values("start_time").reset_index(drop=True)
```

iii. CONVERSION_NOTES.md does not flag any of this as waste; Step 6 frames the double pass as a deliberate
memory trade-off rather than redundant work, and the extra trial columns are described in Step 5 as
serving to *"sanity-check change times and image names"* (which the `--show-processing` plots do). None of
it is large enough to matter against the 396 s total runtime, which is presumably why the AI's own
efficiency review did not revisit it.
